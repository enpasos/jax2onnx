export function parseSpecialNumber(value) {
  if (typeof value === "number") {
    return value;
  }
  if (value === "NaN") {
    return Number.NaN;
  }
  if (value === "Infinity") {
    return Number.POSITIVE_INFINITY;
  }
  if (value === "-Infinity") {
    return Number.NEGATIVE_INFINITY;
  }
  return Number(value);
}

export function float16BitsToNumber(bits) {
  const h = Number(bits) & 0xffff;
  const sign = h & 0x8000 ? -1 : 1;
  const exponent = (h >> 10) & 0x1f;
  const fraction = h & 0x03ff;

  if (exponent === 0) {
    return sign * 2 ** -14 * (fraction / 1024);
  }
  if (exponent === 0x1f) {
    return fraction === 0 ? sign * Number.POSITIVE_INFINITY : Number.NaN;
  }
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

export function makeTypedData(type, data) {
  switch (type) {
    case "float32":
      return Float32Array.from(data, parseSpecialNumber);
    case "float64":
      return Float64Array.from(data, parseSpecialNumber);
    case "float16":
      return Uint16Array.from(data, Number);
    case "uint8":
      return Uint8Array.from(data, Number);
    case "int8":
      return Int8Array.from(data, Number);
    case "uint16":
      return Uint16Array.from(data, Number);
    case "int16":
      return Int16Array.from(data, Number);
    case "uint32":
      return Uint32Array.from(data, Number);
    case "int32":
      return Int32Array.from(data, Number);
    case "int64":
      return BigInt64Array.from(data, BigInt);
    case "uint64":
      return BigUint64Array.from(data, BigInt);
    case "bool":
      return Uint8Array.from(data, (value) => (value ? 1 : 0));
    case "string":
      return data.map(String);
    default:
      throw new Error(`Unsupported tensor type '${type}'.`);
  }
}

export function makeTensor(ort, spec) {
  return new ort.Tensor(spec.type, makeTypedData(spec.type, spec.data), spec.dims);
}

function comparableData(type, data) {
  switch (type) {
    case "float16":
      return Array.from(data, float16BitsToNumber);
    case "float32":
    case "float64":
      return Array.from(data, Number);
    case "int64":
    case "uint64":
      return Array.from(data, BigInt);
    case "bool":
      return Array.from(data, (value) => Boolean(value));
    case "string":
      return Array.from(data, String);
    default:
      return Array.from(data, Number);
  }
}

function comparableSpecData(spec) {
  switch (spec.type) {
    case "float16":
      return spec.data.map(float16BitsToNumber);
    case "float32":
    case "float64":
      return spec.data.map(parseSpecialNumber);
    case "int64":
    case "uint64":
      return spec.data.map(BigInt);
    case "bool":
      return spec.data.map(Boolean);
    case "string":
      return spec.data.map(String);
    default:
      return spec.data.map(Number);
  }
}

export function dimsMatch(actual, expected) {
  if (actual.length !== expected.length) {
    return false;
  }
  return actual.every((dim, index) => Number(dim) === Number(expected[index]));
}

function compareValues(name, actual, expected, rtol, atol) {
  if (actual.type !== expected.type) {
    throw new Error(
      `Output '${name}' type mismatch: wasm=${actual.type} cpu=${expected.type}`,
    );
  }

  if (!dimsMatch(actual.dims, expected.dims)) {
    throw new Error(
      `Output '${name}' shape mismatch: wasm=[${actual.dims.join(", ")}] cpu=[${expected.dims.join(", ")}]`,
    );
  }

  const got = actual.data;
  const want = expected.data;
  if (got.length !== want.length) {
    throw new Error(
      `Output '${name}' element count mismatch: wasm=${got.length} cpu=${want.length}`,
    );
  }

  const floating = ["float16", "float32", "float64"].includes(expected.type);
  let maxDiff = 0;
  for (let i = 0; i < want.length; i += 1) {
    const observed = got[i];
    const reference = want[i];
    if (floating) {
      if (Number.isNaN(reference) && Number.isNaN(observed)) {
        continue;
      }
      if (!Number.isFinite(reference) || !Number.isFinite(observed)) {
        if (Object.is(reference, observed)) {
          continue;
        }
        throw new Error(
          `Output '${name}' mismatch at flat index ${i}: wasm=${observed} cpu=${reference}`,
        );
      }
      const diff = Math.abs(Number(observed) - Number(reference));
      if (diff > maxDiff || Number.isNaN(diff)) {
        maxDiff = diff;
      }
      const limit = atol + rtol * Math.abs(Number(reference));
      if (!(diff <= limit)) {
        throw new Error(
          `Output '${name}' mismatch at flat index ${i}: wasm=${observed} cpu=${reference} diff=${diff} limit=${limit}`,
        );
      }
      continue;
    }

    if (observed !== reference) {
      throw new Error(
        `Output '${name}' mismatch at flat index ${i}: wasm=${String(observed)} cpu=${String(reference)}`,
      );
    }
  }
  return maxDiff;
}

export function compareTensor(name, actual, expected, rtol, atol) {
  return compareValues(
    name,
    {
      type: actual.type,
      dims: actual.dims,
      data: comparableData(actual.type, actual.data),
    },
    {
      type: expected.type,
      dims: expected.dims,
      data: comparableSpecData(expected),
    },
    rtol,
    atol,
  );
}

export function compareTensorSpec(name, actual, expected, rtol, atol) {
  return compareValues(
    name,
    {
      type: actual.type,
      dims: actual.dims,
      data: comparableSpecData(actual),
    },
    {
      type: expected.type,
      dims: expected.dims,
      data: comparableSpecData(expected),
    },
    rtol,
    atol,
  );
}
