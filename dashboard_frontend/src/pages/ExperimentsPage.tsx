import { useEffect, useState } from "react";
import { Alert, CircularProgress, Stack, Typography } from "@mui/material";

import { fetchMetricsComparison } from "../api/metrics";
import { ExperimentTable } from "../components/experiments/ExperimentTable";
import type { ComparisonRow } from "../types/api";

export function ExperimentsPage(): JSX.Element {
  const [rows, setRows] = useState<ComparisonRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const load = async (): Promise<void> => {
      setLoading(true);
      setError(null);
      try {
        setRows(await fetchMetricsComparison());
      } catch (loadError) {
        setError(loadError instanceof Error ? loadError.message : "Failed to load experiments.");
      } finally {
        setLoading(false);
      }
    };
    void load();
  }, []);

  return (
    <Stack spacing={3}>
      <Stack spacing={0.5}>
        <Typography variant="h4" sx={{ fontWeight: 800 }}>
          Experiments
        </Typography>
        <Typography color="text.secondary">
          Unified table of discovered experiments, normalized metrics and artifact status.
        </Typography>
      </Stack>

      {error ? <Alert severity="error">{error}</Alert> : null}
      {loading ? <CircularProgress /> : <ExperimentTable rows={rows} />}
    </Stack>
  );
}
