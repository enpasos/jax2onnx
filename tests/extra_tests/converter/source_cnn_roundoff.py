# tests/extra_tests/converter/source_cnn_roundoff.py

"""CNN-specific exact centers and outward dyadic roundoff envelopes.

Only the standard library is used. This is a finite-sample test oracle, never
an exporter policy or an assertion about an opaque provider's algorithms.
The formula, grid and rounding sites are fixed before observing any outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from math import prod
import struct
import time
from typing import Iterable, Sequence


CRITERION = "cnn-exact-input-forward-roundoff-envelope-v1-outward-grid192-v1"
ERROR_EXPONENT = -192
U = 1 << 24
TAU = 1 << (-126 - ERROR_EXPONENT)
PRODUCTS_PER_IMAGE = 4_643_840
PARAMETER_SHAPES = {
    "conv1.kernel": (3, 3, 1, 32),
    "conv1.bias": (32,),
    "conv2.kernel": (3, 3, 32, 64),
    "conv2.bias": (64,),
    "linear1.kernel": (3136, 256),
    "linear1.bias": (256,),
    "linear2.kernel": (256, 10),
    "linear2.bias": (10,),
}
ASSUMPTIONS = {
    "center": "exact original FP32 dyadics; full original CNN; NHWC flatten",
    "error_grid": "2^-192; upward ceiling once at each affine/pool/ReLU output",
    "affine": "P+gamma(n+1)*S+2*n*tau/(1-(n+1)*u)",
    "pool": "(P+gamma(4)*S+4*tau/(1-4*u)+tau)/4+tau",
    "relu": "clip [c-e-tau,c+e+tau] at zero; outward grid ceiling",
    "fanin": [9, 288, 3136, 256],
    "u": "2^-24",
    "tau": "2^-126; conservative DAZ/FTZ allowance; no caller/session override",
    "provider_algorithms": "UNVERIFIED; observed inequalities only",
    "scope": "recorded finite samples only; no blanket tolerance or public acceptance",
}


def ceil_div(numerator: int, denominator: int) -> int:
    if numerator < 0 or denominator <= 0:
        raise ValueError("ceil_div requires nonnegative numerator and positive divisor")
    return (numerator + denominator - 1) // denominator


def ceil_scale(value: int, source: int, destination: int) -> int:
    if value < 0:
        raise ValueError("outward scaling requires a nonnegative value")
    difference = source - destination
    return value << difference if difference >= 0 else ceil_div(value, 1 << -difference)


def dyadic_le(left: int, left_exp: int, right: int, right_exp: int) -> bool:
    common = min(left_exp, right_exp)
    return (left << (left_exp - common)) <= (right << (right_exp - common))


def fp32_word(word: int, *, normal_or_zero: bool = False) -> tuple[int, int]:
    if not 0 <= word < (1 << 32):
        raise ValueError("invalid FP32 word")
    exponent, fraction = (word >> 23) & 255, word & ((1 << 23) - 1)
    if exponent == 255:
        raise ValueError("nonfinite FP32 value")
    if normal_or_zero and exponent == 0 and fraction:
        raise ValueError("subnormal stored parameter is outside the oracle domain")
    mantissa = fraction if exponent == 0 else (1 << 23) | fraction
    return (-mantissa if word >> 31 else mantissa), (
        -149 if exponent == 0 else exponent - 150
    )


@dataclass(frozen=True)
class Dyadics:
    shape: tuple[int, ...]
    values: tuple[int, ...]
    exponent: int

    def __post_init__(self) -> None:
        if any(type(size) is not int or size <= 0 for size in self.shape):
            raise ValueError("positive concrete tensor dimensions required")
        if prod(self.shape) != len(self.values):
            raise ValueError("dyadic tensor size mismatch")

    @classmethod
    def fp32(
        cls, shape: tuple[int, ...], payload: bytes, *, parameter: bool = False
    ) -> Dyadics:
        if len(payload) != 4 * prod(shape):
            raise ValueError("FP32 tensor payload size mismatch")
        decoded = [
            fp32_word(word[0], normal_or_zero=parameter)
            for word in struct.iter_unpack("<I", payload)
        ]
        exponent = min((power for value, power in decoded if value), default=0)
        return cls(
            shape,
            tuple(
                value << (power - exponent) if value else 0 for value, power in decoded
            ),
            exponent,
        )

    def document(self) -> dict[str, object]:
        return {
            "shape": list(self.shape),
            "exponent": self.exponent,
            "integers": [str(value) for value in self.values],
        }


@dataclass(frozen=True)
class Envelope:
    center: Dyadics
    errors: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.errors) != len(self.center.values) or any(
            value < 0 for value in self.errors
        ):
            raise ValueError("invalid error vector")

    def document(self) -> dict[str, object]:
        return {
            "criterion": CRITERION,
            "assumptions": ASSUMPTIONS,
            "center": self.center.document(),
            "error_exponent": ERROR_EXPONENT,
            "error_integers": [str(value) for value in self.errors],
            "rejects_all_zero": self.rejects_zero(),
        }

    def rejects_zero(self) -> bool:
        return any(
            not dyadic_le(abs(value), self.center.exponent, error, ERROR_EXPONENT)
            for value, error in zip(self.center.values, self.errors, strict=True)
        )

    def check(self, shape: tuple[int, ...], dtype: str, payload: bytes) -> None:
        if dtype != "float32" or shape != self.center.shape:
            raise ValueError(
                "CNN output must have the exact declared FP32 Bx10 signature"
            )
        observed = Dyadics.fp32(shape, payload)
        common = min(observed.exponent, self.center.exponent, ERROR_EXPONENT)
        for index, (actual, center, error) in enumerate(
            zip(observed.values, self.center.values, self.errors, strict=True)
        ):
            distance = abs(
                (actual << (observed.exponent - common))
                - (center << (self.center.exponent - common))
            )
            if distance > error << (ERROR_EXPONENT - common):
                raise ValueError(f"CNN exact envelope mismatch at scalar {index}")

    def check_pair(self, left: bytes, right: bytes) -> None:
        a, b = (Dyadics.fp32(self.center.shape, value) for value in (left, right))
        common = min(a.exponent, b.exponent, ERROR_EXPONENT)
        for index, (x, y, error) in enumerate(
            zip(a.values, b.values, self.errors, strict=True)
        ):
            distance = abs((x << (a.exponent - common)) - (y << (b.exponent - common)))
            if distance > (2 * error) << (ERROR_EXPONENT - common):
                raise ValueError(f"CNN source/target pair mismatch at scalar {index}")


@dataclass
class Budget:
    max_products: int
    max_seconds: float = 180.0
    max_integer_bits: int = 8192
    products: int = 0
    started: float = field(default_factory=time.monotonic)
    stages: list[dict[str, object]] = field(default_factory=list)

    def step(self, products: int = 0) -> None:
        self.products += products
        if (
            self.products > self.max_products
            or time.monotonic() - self.started > self.max_seconds
        ):
            raise ValueError("CNN oracle setup exceeded its frozen compute budget")

    def stage(self, name: str, value: Envelope) -> None:
        self.step()
        bits = max(
            (abs(item).bit_length() for item in (*value.center.values, *value.errors)),
            default=0,
        )
        if bits > self.max_integer_bits:
            raise ValueError("CNN oracle integer budget exceeded")
        self.stages.append(
            {
                "stage": name,
                "shape": list(value.center.shape),
                "products": self.products,
                "wall_seconds": time.monotonic() - self.started,
                "maximum_integer_bits": bits,
            }
        )


def overflow_gate(magnitude: int, exponent: int, rounds: int, events: int) -> None:
    """Strictly bound every intermediate below max finite, using upper S."""
    if not 0 < rounds < U:
        raise ValueError("invalid gamma domain")
    tau = ceil_scale(1, -126, exponent)
    left = (magnitude + events * tau) * U
    right = (U - rounds) * ((1 << 24) - 1)
    # maximum finite = (2^24-1)*2^104. Equality is rejected.
    if dyadic_le(right, 104, left, exponent):
        raise ValueError("CNN roundoff model cannot exclude FP32 intermediate overflow")


def affine(
    incoming: Envelope,
    weights: Dyadics,
    bias: Dyadics,
    indices: Iterable[Sequence[int | None]],
    output_shape: tuple[int, ...],
    fanin: int,
    budget: Budget,
) -> Envelope:
    """HWIO/IO output-channel-last weights; None denotes literal padding zero."""
    width = weights.shape[-1]
    if len(weights.values) != fanin * width or bias.shape != (width,):
        raise ValueError("affine original parameter geometry mismatch")
    c = incoming.center
    bound_exp = min(
        ERROR_EXPONENT,
        ERROR_EXPONENT + weights.exponent,
        c.exponent + weights.exponent,
        bias.exponent,
    )
    center_exp = min(c.exponent + weights.exponent, bias.exponent)
    center_shift = c.exponent + weights.exponent - center_exp
    bias_shift = bias.exponent - center_exp
    errors = incoming.errors
    rounds, denominator = fanin + 1, U - fanin - 1
    if denominator <= 0:
        raise ValueError("affine gamma domain failed")
    bound_tau = 1 << (-126 - bound_exp)
    centers_out, errors_out = [], []
    for locations in indices:
        if len(locations) != fanin:
            raise ValueError("affine fanin mismatch")
        for channel in range(width):
            center = propagated = magnitude = 0
            for term, location in enumerate(locations):
                weight = weights.values[term * width + channel]
                # Padding participates in all frozen tau terms and full n.
                value = 0 if location is None else c.values[location]
                error = TAU if location is None else errors[location] + TAU
                center += weight * value
                propagated += abs(weight) * error
                magnitude += abs(weight) * abs(value)
            propagated <<= weights.exponent + ERROR_EXPONENT - bound_exp
            magnitude = (
                propagated
                + (magnitude << (c.exponent + weights.exponent - bound_exp))
                + (abs(bias.values[channel]) << (bias.exponent - bound_exp))
            )
            overflow_gate(magnitude, bound_exp, rounds, 2 * fanin)
            numerator = (
                denominator * propagated
                + rounds * magnitude
                + 2 * fanin * bound_tau * U
            )
            error = ceil_div(numerator, denominator << (ERROR_EXPONENT - bound_exp))
            centers_out.append(
                (center << center_shift) + (bias.values[channel] << bias_shift)
            )
            errors_out.append(error)
        budget.step(fanin * width)
    return Envelope(
        Dyadics(output_shape, tuple(centers_out), center_exp), tuple(errors_out)
    )


def relu(incoming: Envelope) -> Envelope:
    c = incoming.center
    common = min(c.exponent, ERROR_EXPONENT)
    centers, errors = [], []
    for value, error in zip(c.values, incoming.errors, strict=True):
        center = value << (c.exponent - common)
        radius = (error + TAU) << (ERROR_EXPONENT - common)
        clipped = max(center, 0)
        lower, upper = max(0, center - radius), max(0, center + radius)
        centers.append(max(value, 0))
        errors.append(
            ceil_scale(max(clipped - lower, upper - clipped), common, ERROR_EXPONENT)
        )
    return Envelope(Dyadics(c.shape, tuple(centers), c.exponent), tuple(errors))


def pool(incoming: Envelope) -> Envelope:
    batch, height, width, channels = incoming.center.shape
    if height % 2 or width % 2:
        raise ValueError("original disjoint pool geometry required")
    c, e = incoming.center, incoming.errors
    centers, errors = [], []
    denominator = U - 4
    bound_exp = min(c.exponent, ERROR_EXPONENT)
    bound_tau = 1 << (-126 - bound_exp)
    for image in range(batch):
        for row in range(0, height, 2):
            for column in range(0, width, 2):
                for channel in range(channels):
                    locations = [
                        ((image * height + row + dr) * width + column + dc) * channels
                        + channel
                        for dr, dc in ((0, 0), (0, 1), (1, 0), (1, 1))
                    ]
                    propagated = sum(e[index] + TAU for index in locations) << (
                        ERROR_EXPONENT - bound_exp
                    )
                    magnitude = propagated + (
                        sum(abs(c.values[index]) for index in locations)
                        << (c.exponent - bound_exp)
                    )
                    overflow_gate(magnitude, bound_exp, 4, 4)
                    # One ceiling of the complete pool bound; sum then Div4.
                    numerator = (
                        denominator * propagated
                        + 4 * magnitude
                        + (4 * U + 5 * denominator) * bound_tau
                    )
                    errors.append(
                        ceil_div(
                            numerator, (4 * denominator) << (ERROR_EXPONENT - bound_exp)
                        )
                    )
                    centers.append(sum(c.values[index] for index in locations))
    return Envelope(
        Dyadics(
            (batch, height // 2, width // 2, channels), tuple(centers), c.exponent - 2
        ),
        tuple(errors),
    )


def _conv_locations(shape: tuple[int, ...]) -> Iterable[tuple[int | None, ...]]:
    batch, height, width, channels = shape
    for image in range(batch):
        for row in range(height):
            for column in range(width):
                yield tuple(
                    (
                        (
                            ((image * height + row + dr) * width + column + dc)
                            * channels
                            + channel
                        )
                        if 0 <= row + dr < height and 0 <= column + dc < width
                        else None
                    )
                    for dr in (-1, 0, 1)
                    for dc in (-1, 0, 1)
                    for channel in range(channels)
                )


def original_cnn(
    parameters: dict[str, Dyadics], feed: Dyadics, budget: Budget
) -> Envelope:
    if set(parameters) != set(PARAMETER_SHAPES) or any(
        parameters[name].shape != shape for name, shape in PARAMETER_SHAPES.items()
    ):
        raise ValueError("all eight original full parameter tensors required")
    if feed.shape[1:] != (28, 28, 1):
        raise ValueError("original CNN input geometry required")
    batch = feed.shape[0]
    value = Envelope(feed, (0,) * len(feed.values))
    for name, channels, fanin in (("conv1", 32, 9), ("conv2", 64, 288)):
        b, h, w, _ = value.center.shape
        value = affine(
            value,
            parameters[f"{name}.kernel"],
            parameters[f"{name}.bias"],
            _conv_locations(value.center.shape),
            (b, h, w, channels),
            fanin,
            budget,
        )
        budget.stage(name, value)
        value = pool(relu(value))
        budget.stage(name + ".relu_sum4_div4", value)
    value = Envelope(
        Dyadics((batch, 3136), value.center.values, value.center.exponent), value.errors
    )
    for name, fanin, channels in (("linear1", 3136, 256), ("linear2", 256, 10)):
        locations = (
            range(image * fanin, (image + 1) * fanin) for image in range(batch)
        )
        value = affine(
            value,
            parameters[f"{name}.kernel"],
            parameters[f"{name}.bias"],
            locations,
            (batch, channels),
            fanin,
            budget,
        )
        budget.stage(name, value)
        if name == "linear1":
            value = relu(value)
    if budget.products != batch * PRODUCTS_PER_IMAGE:
        raise ValueError("original CNN full product count changed")
    return value


def identity(
    parameters: dict[str, tuple[tuple[int, ...], bytes]],
    feed: tuple[tuple[int, ...], bytes],
) -> str:
    digest = hashlib.sha256(
        json.dumps(
            {"criterion": CRITERION, "assumptions": ASSUMPTIONS}, sort_keys=True
        ).encode()
    )
    for name, (shape, payload) in (*sorted(parameters.items()), ("feed", feed)):
        digest.update(json.dumps([name, shape], separators=(",", ":")).encode())
        digest.update(struct.pack("<Q", len(payload)))
        digest.update(payload)
    return digest.hexdigest()
