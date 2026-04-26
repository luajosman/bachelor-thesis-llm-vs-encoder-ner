export type ExperimentStatus = "not_started" | "trained" | "inferred" | "complete" | "failed";
export type ExperimentModelType = "encoder" | "llm";
export type ExperimentRegime = "encoder" | "zeroshot" | "lora";
export type JobStatus = "queued" | "running" | "completed" | "failed" | "cancelled";

export type ActionName =
  | "run_all"
  | "encoder_only"
  | "zeroshot_only"
  | "lora_only"
  | "deberta_base"
  | "deberta_large"
  | "qwen35_08b_zs"
  | "qwen35_08b_lora"
  | "qwen35_4b_zs"
  | "qwen35_4b_lora"
  | "qwen35_27b_zs"
  | "qwen35_27b_lora"
  | "compare_only";

export interface ExperimentSummary {
  id: string;
  experiment_name: string;
  model_name: string;
  model_type: ExperimentModelType;
  regime: ExperimentRegime;
  dataset: string;
  status: ExperimentStatus;
  output_dir: string;
  has_results: boolean;
  has_inference_metrics: boolean;
  has_predictions: boolean;
  has_best_model: boolean;
  has_best_lora_adapter: boolean;
}

export interface ExperimentMetrics {
  test_f1: number | null;
  test_precision: number | null;
  test_recall: number | null;
  latency_ms_mean: number | null;
  latency_ms_p95: number | null;
  vram_peak_mb: number | null;
  total_params: number | null;
  trainable_params: number | null;
  train_runtime_seconds: number | null;
  best_dev_f1: number | null;
  best_epoch: number | null;
  parse_failure_rate: number | null;
}

export interface PredictionSample {
  sample_id: string;
  tokens: string[];
  gold_bio: string[];
  pred_bio: string[];
  gold_entities: Record<string, unknown>[];
  pred_entities: Record<string, unknown>[];
  raw_output: string | null;
  parse_status: string | null;
}

export interface Job {
  job_id: string;
  action: string;
  status: JobStatus;
  started_at: string | null;
  finished_at: string | null;
  duration: number | null;
  command_label: string;
  exit_code: number | null;
  log_path: string;
}

export interface SystemInfo {
  python_version: string;
  cuda_available: boolean;
  gpu_name: string | null;
  vram_total_mb: number | null;
}

export interface ConfigSummary {
  id: string;
  name: string;
  path: string;
  experiment_name: string;
  model_name: string;
  model_type: ExperimentModelType;
  regime: ExperimentRegime;
  dataset: string;
  output_dir: string;
}

export interface ConfigDetail extends ConfigSummary {
  config: Record<string, unknown>;
}

export interface ArtifactSummary {
  name: string;
  path: string;
  kind: "file" | "directory";
  mime_type: string | null;
  size_bytes: number | null;
  exists: boolean;
}

export interface ArtifactContent {
  artifact: ArtifactSummary;
  content_type: "json" | "yaml" | "text" | "directory" | "binary";
  content: unknown;
  children: ArtifactSummary[];
}

export interface ExperimentDetail {
  summary: ExperimentSummary;
  config: ConfigSummary | null;
  artifacts: ArtifactSummary[];
}

export interface ComparisonRow {
  experiment_id: string;
  experiment_name: string;
  model_name: string;
  model_type: ExperimentModelType;
  regime: ExperimentRegime;
  dataset: string;
  status: ExperimentStatus;
  metrics: ExperimentMetrics;
}

export interface MetricsSummary {
  best_f1: number | null;
  best_experiment_id: string | null;
  best_encoder_id: string | null;
  best_llm_id: string | null;
  fastest_experiment_id: string | null;
  lowest_vram_experiment_id: string | null;
  completed_count: number;
  missing_count: number;
  running_count: number;
}

export interface PerEntityMetricsResponse {
  experiment_id: string;
  entity_types: string[];
  metrics: Record<string, { precision: number; recall: number; f1: number; support: number }>;
}

export interface PerEntityComparisonItem {
  experiment_id: string;
  experiment_name: string;
  regime: ExperimentRegime;
  dataset: string;
  metrics: Record<string, { precision: number; recall: number; f1: number; support: number }>;
}

export interface PerEntityComparisonResponse {
  entity_types: string[];
  experiments: PerEntityComparisonItem[];
}

export interface ErrorAnalysisResponse {
  experiment_id: string;
  summary: Record<string, number | string>;
  examples: Record<string, Record<string, unknown>[]>;
}

export interface ErrorComparisonResponse {
  encoder_id: string | null;
  llm_id: string | null;
  encoder: ErrorAnalysisResponse | null;
  llm: ErrorAnalysisResponse | null;
}

export interface JobLogsResponse {
  job: Job;
  log: string;
}
