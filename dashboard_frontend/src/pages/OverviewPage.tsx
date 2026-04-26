import { useEffect, useMemo, useState } from "react";
import {
  Alert,
  CircularProgress,
  Grid,
  Stack,
  Typography,
} from "@mui/material";

import { fetchJobs } from "../api/jobs";
import { fetchMetricsComparison, fetchMetricsSummary, fetchSystemInfo } from "../api/metrics";
import { formatMegabytes, formatMetric, formatMilliseconds } from "../app/utils";
import { F1BarChart } from "../components/charts/F1BarChart";
import { RegimeDistributionChart } from "../components/charts/RegimeDistributionChart";
import { ScatterMetricChart } from "../components/charts/ScatterMetricChart";
import { MetricCard } from "../components/metrics/MetricCard";
import type { ComparisonRow, Job, MetricsSummary, SystemInfo } from "../types/api";

export function OverviewPage(): JSX.Element {
  const [summary, setSummary] = useState<MetricsSummary | null>(null);
  const [rows, setRows] = useState<ComparisonRow[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [system, setSystem] = useState<SystemInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const load = async (): Promise<void> => {
      setLoading(true);
      setError(null);
      try {
        const [summaryData, comparisonData, jobData, systemData] = await Promise.all([
          fetchMetricsSummary(),
          fetchMetricsComparison(),
          fetchJobs(),
          fetchSystemInfo(),
        ]);
        setSummary(summaryData);
        setRows(comparisonData);
        setJobs(jobData);
        setSystem(systemData);
      } catch (loadError) {
        setError(loadError instanceof Error ? loadError.message : "Failed to load dashboard overview.");
      } finally {
        setLoading(false);
      }
    };
    void load();
  }, []);

  const lookup = useMemo(() => new Map(rows.map((row) => [row.experiment_id, row])), [rows]);
  const runningJobs = jobs.filter((job) => job.status === "queued" || job.status === "running").length;

  if (loading) {
    return <CircularProgress />;
  }

  return (
    <Stack spacing={3}>
      <Stack spacing={0.5}>
        <Typography variant="h4" sx={{ fontWeight: 800 }}>
          Overview
        </Typography>
        <Typography color="text.secondary">
          Summary metrics, runtime health and comparison charts for the current experiment set.
        </Typography>
      </Stack>

      {error ? <Alert severity="error">{error}</Alert> : null}

      <Grid container spacing={2}>
        <Grid item xs={12} md={4} lg={2}>
          <MetricCard
            title="Best F1"
            value={formatMetric(summary?.best_f1)}
            subtitle={summary?.best_experiment_id ?? "No completed experiment"}
          />
        </Grid>
        <Grid item xs={12} md={4} lg={2}>
          <MetricCard
            title="Best Encoder"
            value={lookup.get(summary?.best_encoder_id ?? "")?.experiment_name ?? "—"}
            subtitle={formatMetric(lookup.get(summary?.best_encoder_id ?? "")?.metrics.test_f1)}
          />
        </Grid>
        <Grid item xs={12} md={4} lg={2}>
          <MetricCard
            title="Best LLM"
            value={lookup.get(summary?.best_llm_id ?? "")?.experiment_name ?? "—"}
            subtitle={formatMetric(lookup.get(summary?.best_llm_id ?? "")?.metrics.test_f1)}
          />
        </Grid>
        <Grid item xs={12} md={4} lg={2}>
          <MetricCard
            title="Fastest Model"
            value={lookup.get(summary?.fastest_experiment_id ?? "")?.experiment_name ?? "—"}
            subtitle={formatMilliseconds(lookup.get(summary?.fastest_experiment_id ?? "")?.metrics.latency_ms_mean)}
          />
        </Grid>
        <Grid item xs={12} md={4} lg={2}>
          <MetricCard
            title="Lowest VRAM"
            value={lookup.get(summary?.lowest_vram_experiment_id ?? "")?.experiment_name ?? "—"}
            subtitle={formatMegabytes(lookup.get(summary?.lowest_vram_experiment_id ?? "")?.metrics.vram_peak_mb)}
          />
        </Grid>
        <Grid item xs={12} md={4} lg={2}>
          <MetricCard
            title="Counts"
            value={`${summary?.completed_count ?? 0} complete`}
            subtitle={`${summary?.missing_count ?? 0} missing · ${runningJobs} running`}
          />
        </Grid>
      </Grid>

      {system ? (
        <Alert severity={system.cuda_available ? "info" : "warning"}>
          Python {system.python_version}
          {system.cuda_available && system.gpu_name ? ` · ${system.gpu_name}` : " · CUDA unavailable"}
          {system.vram_total_mb ? ` · ${(system.vram_total_mb / 1024).toFixed(1)} GB VRAM` : ""}
        </Alert>
      ) : null}

      <Grid container spacing={2}>
        <Grid item xs={12} lg={8}>
          <F1BarChart rows={rows} />
        </Grid>
        <Grid item xs={12} lg={4}>
          <RegimeDistributionChart rows={rows} />
        </Grid>
        <Grid item xs={12} md={6}>
          <ScatterMetricChart
            rows={rows}
            title="F1 vs Latency"
            xKey="latency_ms_mean"
            yKey="test_f1"
            xLabel="Latency (ms)"
            yLabel="F1"
          />
        </Grid>
        <Grid item xs={12} md={6}>
          <ScatterMetricChart
            rows={rows}
            title="F1 vs VRAM"
            xKey="vram_peak_mb"
            yKey="test_f1"
            xLabel="VRAM (MB)"
            yLabel="F1"
          />
        </Grid>
      </Grid>
    </Stack>
  );
}
