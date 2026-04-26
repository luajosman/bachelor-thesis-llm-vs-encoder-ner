import { useEffect, useState } from "react";
import { Alert, CircularProgress, Grid, Stack, Typography } from "@mui/material";

import { fetchMetricsComparison, fetchPerEntityComparison } from "../api/metrics";
import { F1BarChart } from "../components/charts/F1BarChart";
import { Heatmap } from "../components/charts/Heatmap";
import { PrecisionRecallChart } from "../components/charts/PrecisionRecallChart";
import { ScatterMetricChart } from "../components/charts/ScatterMetricChart";
import type { ComparisonRow, PerEntityComparisonResponse } from "../types/api";

export function ComparePage(): JSX.Element {
  const [rows, setRows] = useState<ComparisonRow[]>([]);
  const [perEntity, setPerEntity] = useState<PerEntityComparisonResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const load = async (): Promise<void> => {
      setLoading(true);
      setError(null);
      try {
        const [comparisonData, entityData] = await Promise.all([
          fetchMetricsComparison(),
          fetchPerEntityComparison(),
        ]);
        setRows(comparisonData);
        setPerEntity(entityData);
      } catch (loadError) {
        setError(loadError instanceof Error ? loadError.message : "Failed to load comparison data.");
      } finally {
        setLoading(false);
      }
    };
    void load();
  }, []);

  if (loading) {
    return <CircularProgress />;
  }

  return (
    <Stack spacing={3}>
      <Stack spacing={0.5}>
        <Typography variant="h4" sx={{ fontWeight: 800 }}>
          Compare Models
        </Typography>
        <Typography color="text.secondary">
          Cross-model comparisons for final test metrics, efficiency and per-entity behavior.
        </Typography>
      </Stack>

      {error ? <Alert severity="error">{error}</Alert> : null}

      <Grid container spacing={2}>
        <Grid item xs={12}>
          <F1BarChart rows={rows} title="Final Test F1 Across Models" />
        </Grid>
        <Grid item xs={12}>
          <PrecisionRecallChart rows={rows} />
        </Grid>
        <Grid item xs={12} md={6}>
          <ScatterMetricChart rows={rows} title="VRAM vs F1" xKey="vram_peak_mb" yKey="test_f1" xLabel="VRAM (MB)" yLabel="F1" />
        </Grid>
        <Grid item xs={12} md={6}>
          <ScatterMetricChart rows={rows} title="Latency vs F1" xKey="latency_ms_mean" yKey="test_f1" xLabel="Latency (ms)" yLabel="F1" />
        </Grid>
        <Grid item xs={12}>
          <ScatterMetricChart rows={rows} title="Params vs F1" xKey="total_params" yKey="test_f1" xLabel="Total Params" yLabel="F1" />
        </Grid>
        {perEntity ? (
          <Grid item xs={12}>
            <Heatmap data={perEntity} />
          </Grid>
        ) : null}
      </Grid>
    </Stack>
  );
}
