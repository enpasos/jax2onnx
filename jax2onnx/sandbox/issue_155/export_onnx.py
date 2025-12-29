#!/usr/bin/env python
# jax2onnx/sandbox/issue_155/export_onnx.py

# Import dependencies
import warnings
import argparse
import inspect
import json
from pathlib import Path

# Import jax and brax
import jax
import jax.numpy as jp
import numpy as np
from brax.training import checkpoint
from brax.training.agents.ppo import networks as ppo_networks
from jax2onnx import to_onnx


# Main
def main():
    # Parse arguments
    args = parse_args()

    # Resolve load checkpoint path
    ckpt = Path(args.load).resolve()
    if ckpt.is_file():
        ckpt = ckpt.parent
    if ckpt.name != "checkpoints" and (ckpt / "checkpoints").exists():
        ckpt = ckpt / "checkpoints"

    # Resolve output path
    out_path = Path(args.save) if args.save else ckpt / "policy.onnx"

    # Export model
    export_onnx(ckpt, out_path)


# Parse args
def parse_args():
    # Build parser and parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", required=True, help="Path to checkpoint dir or file")
    parser.add_argument("--save", default=None, help="Output ONNX path")
    return parser.parse_args()


# Export via jax2onnx
def export_onnx(ckpt_dir: Path, out_path: Path):
    # Build policy
    policy_fn, obs_dim, act_dim = build_policy(ckpt_dir)

    # Define single step policy
    def policy_single(obs):
        obs = jp.reshape(obs, (1, obs_dim))
        action, _ = policy_fn(obs, jax.random.PRNGKey(0))
        return action[0]

    # Make shape placeholder
    sample = jax.ShapeDtypeStruct((1, obs_dim), jp.float32)

    # Convert to onnx
    model = to_onnx(
        policy_single,
        (sample,),
        return_mode="file",
        output_path=str(out_path),
    )

    # Print
    print(f"Saved ONNX to {out_path} (inputs: [1, {obs_dim}], outputs: [1, {act_dim}])")

    # Return model
    return model


def _resolve_checkpoint_dir(checkpoint_dir: Path) -> Path:
    if any(checkpoint_dir.glob("*.json")):
        return checkpoint_dir

    candidates = []
    for entry in sorted(checkpoint_dir.iterdir()):
        if entry.is_dir() and any(entry.glob("*.json")):
            candidates.append(entry)

    if not candidates:
        return checkpoint_dir

    numeric = [entry for entry in candidates if entry.name.isdigit()]
    if numeric:
        return sorted(numeric, key=lambda p: p.name)[-1]
    return sorted(candidates, key=lambda p: p.name)[-1]


# Build policy fn and shapes
def build_policy(checkpoint_dir: Path):
    checkpoint_dir = _resolve_checkpoint_dir(checkpoint_dir)

    # Read configs
    config = load_checkpoint_config(checkpoint_dir)
    network_config = prepare_network_config(config.get("network_factory_kwargs", {}))
    obs_config = config.get("observation_size")

    # Get observation size
    if isinstance(obs_config, dict) and "shape" in obs_config:
        observation_size = int(np.prod(tuple(obs_config["shape"])))
    elif isinstance(obs_config, int):
        observation_size = obs_config
    else:
        raise ValueError("Checkpoint missing observation_size metadata")

    # Get action size
    action_size = int(config.get("action_size", 0))
    if action_size <= 0:
        raise ValueError("Checkpoint missing action_size metadata")

    # Create networks
    networks = ppo_networks.make_ppo_networks(
        observation_size=observation_size, action_size=action_size, **network_config
    )

    # Load params (normalizer state, policy params)
    loaded = checkpoint.load(checkpoint_dir)
    params = (loaded[0], loaded[1])

    # Build deterministic policy
    make_inference_fn = ppo_networks.make_inference_fn(networks)
    policy_fn = make_inference_fn(params, deterministic=True)

    # Return policy, observation size, and action size
    return policy_fn, observation_size, action_size


# Read network config
def load_checkpoint_config(checkpoint_dir: Path):
    # Any json file in the directory
    for candidate in checkpoint_dir.glob("*.json"):
        with candidate.open("r") as file:
            return json.load(file)

    # No config found
    raise FileNotFoundError(f"Missing config json in {checkpoint_dir}")


# Prepare network args
def prepare_network_config(network_config):
    # Normalize network args
    config = dict(network_config)
    if "policy_hidden_layer_sizes" in config:
        config["policy_hidden_layer_sizes"] = tuple(config["policy_hidden_layer_sizes"])
    if "value_hidden_layer_sizes" in config:
        config["value_hidden_layer_sizes"] = tuple(config["value_hidden_layer_sizes"])
    valid = set(inspect.signature(ppo_networks.make_ppo_networks).parameters)
    for key in list(config.keys()):
        if key not in valid:
            config.pop(key, None)

    # Build activation map
    activation_map = {
        "relu": jax.nn.relu,
        "tanh": jax.nn.tanh,
        "sigmoid": jax.nn.sigmoid,
        "swish": jax.nn.swish,
        "silu": jax.nn.silu,
        "gelu": jax.nn.gelu,
        "elu": jax.nn.elu,
    }

    # Build kernel inits
    kernel_inits = {
        "lecun_uniform": jax.nn.initializers.lecun_uniform,
        "lecun_normal": jax.nn.initializers.lecun_normal,
        "glorot_uniform": jax.nn.initializers.glorot_uniform,
        "glorot_normal": jax.nn.initializers.glorot_normal,
        "kaiming_uniform": jax.nn.initializers.kaiming_uniform,
        "kaiming_normal": jax.nn.initializers.kaiming_normal,
        "variance_scaling": jax.nn.initializers.variance_scaling,
    }

    # Normalize activation and kernel inits
    for key, value in list(config.items()):
        if isinstance(value, str) and "activation" in key:
            config[key] = activation_map.get(value.lower(), value)
        if isinstance(value, str) and "kernel_init" in key:
            config[key] = kernel_inits.get(value.lower(), value)

    # Return normalized config
    return config


# Silence float64 truncation warnings
warnings.filterwarnings(
    "ignore", message="Explicitly requested dtype float64 requested"
)


# Run main
if __name__ == "__main__":
    main()
