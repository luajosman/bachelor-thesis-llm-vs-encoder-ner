import { apiClient } from "./client";
import type { ComparisonRow, ErrorComparisonResponse, MetricsSummary, PerEntityComparisonResponse, SystemInfo } from "../types/api";

export async function fetchSystemInfo(): Promise<SystemInfo> {
  const { data } = await apiClient.get<SystemInfo>("/system");
  return data;
}

export async function fetchMetricsSummary(dataset = "multinerd"): Promise<MetricsSummary> {
  const { data } = await apiClient.get<MetricsSummary>("/metrics/summary", {
    params: { dataset },
  });
  return data;
}

export async function fetchMetricsComparison(dataset = "multinerd"): Promise<ComparisonRow[]> {
  const { data } = await apiClient.get<ComparisonRow[]>("/metrics/comparison", {
    params: { dataset },
  });
  return data;
}

export async function fetchPerEntityComparison(dataset = "multinerd"): Promise<PerEntityComparisonResponse> {
  const { data } = await apiClient.get<PerEntityComparisonResponse>("/metrics/per-entity-comparison", {
    params: { dataset },
  });
  return data;
}

export async function fetchErrorComparison(
  dataset = "multinerd",
  encoderId?: string,
  llmId?: string,
): Promise<ErrorComparisonResponse> {
  const { data } = await apiClient.get<ErrorComparisonResponse>("/error-analysis/compare", {
    params: { dataset, encoder_id: encoderId, llm_id: llmId },
  });
  return data;
}
