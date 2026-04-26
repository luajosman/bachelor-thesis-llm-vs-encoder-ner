import { apiClient } from "./client";
import type {
  ArtifactContent,
  ArtifactSummary,
  ConfigDetail,
  ConfigSummary,
  ErrorAnalysisResponse,
  ExperimentDetail,
  ExperimentMetrics,
  ExperimentSummary,
  PerEntityMetricsResponse,
  PredictionSample,
} from "../types/api";

export async function fetchConfigs(): Promise<ConfigSummary[]> {
  const { data } = await apiClient.get<ConfigSummary[]>("/configs");
  return data;
}

export async function fetchConfig(configId: string): Promise<ConfigDetail> {
  const { data } = await apiClient.get<ConfigDetail>(`/configs/${configId}`);
  return data;
}

export async function fetchExperiments(dataset = "multinerd"): Promise<ExperimentSummary[]> {
  const { data } = await apiClient.get<ExperimentSummary[]>("/experiments", {
    params: { dataset },
  });
  return data;
}

export async function fetchExperiment(experimentId: string, dataset = "multinerd"): Promise<ExperimentDetail> {
  const { data } = await apiClient.get<ExperimentDetail>(`/experiments/${experimentId}`, {
    params: { dataset },
  });
  return data;
}

export async function fetchExperimentMetrics(experimentId: string, dataset = "multinerd"): Promise<ExperimentMetrics> {
  const { data } = await apiClient.get<ExperimentMetrics>(`/experiments/${experimentId}/metrics`, {
    params: { dataset },
  });
  return data;
}

export async function fetchExperimentPredictions(experimentId: string, dataset = "multinerd"): Promise<PredictionSample[]> {
  const { data } = await apiClient.get<PredictionSample[]>(`/experiments/${experimentId}/predictions`, {
    params: { dataset },
  });
  return data;
}

export async function fetchPredictionSample(experimentId: string, sampleId: string, dataset = "multinerd"): Promise<PredictionSample> {
  const { data } = await apiClient.get<PredictionSample>(`/experiments/${experimentId}/predictions/${sampleId}`, {
    params: { dataset },
  });
  return data;
}

export async function fetchExperimentErrors(experimentId: string, dataset = "multinerd"): Promise<ErrorAnalysisResponse> {
  const { data } = await apiClient.get<ErrorAnalysisResponse>(`/experiments/${experimentId}/errors`, {
    params: { dataset },
  });
  return data;
}

export async function fetchExperimentPerEntity(experimentId: string, dataset = "multinerd"): Promise<PerEntityMetricsResponse> {
  const { data } = await apiClient.get<PerEntityMetricsResponse>(`/experiments/${experimentId}/per-entity`, {
    params: { dataset },
  });
  return data;
}

export async function fetchArtifacts(experimentId: string, dataset = "multinerd"): Promise<ArtifactSummary[]> {
  const { data } = await apiClient.get<ArtifactSummary[]>(`/experiments/${experimentId}/artifacts`, {
    params: { dataset },
  });
  return data;
}

export async function fetchArtifactContent(experimentId: string, artifactName: string, dataset = "multinerd"): Promise<ArtifactContent> {
  const { data } = await apiClient.get<ArtifactContent>(`/experiments/${experimentId}/artifacts/${artifactName}`, {
    params: { dataset },
  });
  return data;
}
