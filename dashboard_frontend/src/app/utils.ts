export function formatMetric(value: number | null | undefined, digits = 3): string {
  return value === null || value === undefined || Number.isNaN(value) ? "—" : value.toFixed(digits);
}

export function formatMilliseconds(value: number | null | undefined): string {
  return value === null || value === undefined ? "—" : `${value.toFixed(1)} ms`;
}

export function formatMegabytes(value: number | null | undefined): string {
  return value === null || value === undefined ? "—" : `${(value / 1024).toFixed(1)} GB`;
}

export function formatParams(value: number | null | undefined): string {
  if (value === null || value === undefined) {
    return "—";
  }
  if (value >= 1_000_000_000) {
    return `${(value / 1_000_000_000).toFixed(1)}B`;
  }
  if (value >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(0)}M`;
  }
  return value.toString();
}
