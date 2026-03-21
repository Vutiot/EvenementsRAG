/** Shared config merge helpers — used by QueryTester and BenchmarkRuns. */

/** Deep merge overrides into base (immutable — returns new object). */
export function deepMerge(
  base: Record<string, unknown>,
  overrides: Record<string, unknown>,
): Record<string, unknown> {
  const result = { ...base };
  for (const key of Object.keys(overrides)) {
    const bv = base[key];
    const ov = overrides[key];
    if (
      bv != null &&
      ov != null &&
      typeof bv === "object" &&
      !Array.isArray(bv) &&
      typeof ov === "object" &&
      !Array.isArray(ov)
    ) {
      result[key] = deepMerge(
        bv as Record<string, unknown>,
        ov as Record<string, unknown>,
      );
    } else {
      result[key] = ov;
    }
  }
  return result;
}

/** Set a dotted path in a nested override object. Removes path if value is undefined. */
export function setOverridePath(
  overrides: Record<string, unknown>,
  path: string,
  value: unknown,
): Record<string, unknown> {
  const parts = path.split(".");
  const head = parts[0] as string;
  if (parts.length === 1) {
    const next = { ...overrides };
    if (value === undefined) {
      delete next[head];
    } else {
      next[head] = value;
    }
    return next;
  }

  const rest = parts.slice(1).join(".");
  const child = (overrides[head] ?? {}) as Record<string, unknown>;
  const updated = setOverridePath(child, rest, value);

  const next = { ...overrides };
  // Remove empty sub-objects
  if (Object.keys(updated).length === 0) {
    delete next[head];
  } else {
    next[head] = updated;
  }
  return next;
}

/** Count leaf values in a nested override object. */
export function countOverrides(obj: Record<string, unknown>): number {
  let count = 0;
  for (const v of Object.values(obj)) {
    if (v != null && typeof v === "object" && !Array.isArray(v)) {
      count += countOverrides(v as Record<string, unknown>);
    } else {
      count += 1;
    }
  }
  return count;
}

/** Compute cartesian product size from overrides that contain arrays. */
export function computeSweepCombinations(overrides: Record<string, unknown>): number {
  let product = 1;
  function walk(obj: Record<string, unknown>) {
    for (const v of Object.values(obj)) {
      if (Array.isArray(v) && v.length > 1) {
        product *= v.length;
      } else if (v != null && typeof v === "object" && !Array.isArray(v)) {
        walk(v as Record<string, unknown>);
      }
    }
  }
  walk(overrides);
  return product;
}

/** Split overrides into sweep_params (array values) and scalar config_overrides. */
export function extractSweepParams(overrides: Record<string, unknown>): {
  sweepParams: Record<string, unknown[]>;
  configOverrides: Record<string, unknown>;
} {
  const sweepParams: Record<string, unknown[]> = {};
  const configOverrides: Record<string, unknown> = {};

  function walk(obj: Record<string, unknown>, prefix: string) {
    for (const [key, val] of Object.entries(obj)) {
      const path = prefix ? `${prefix}.${key}` : key;
      if (Array.isArray(val)) {
        if (val.length > 1) {
          sweepParams[path] = val;
        } else if (val.length === 1) {
          configOverrides[path] = val[0];
        }
      } else if (val != null && typeof val === "object") {
        walk(val as Record<string, unknown>, path);
      } else {
        configOverrides[path] = val;
      }
    }
  }
  walk(overrides, "");

  // Restructure configOverrides from flat dotted paths to nested object
  const nested: Record<string, unknown> = {};
  for (const [path, val] of Object.entries(configOverrides)) {
    const parts = path.split(".");
    let cur: Record<string, unknown> = nested;
    for (let i = 0; i < parts.length - 1; i++) {
      if (!(parts[i] as string in cur)) cur[parts[i] as string] = {};
      cur = cur[parts[i] as string] as Record<string, unknown>;
    }
    cur[parts[parts.length - 1] as string] = val;
  }

  return { sweepParams, configOverrides: nested };
}
