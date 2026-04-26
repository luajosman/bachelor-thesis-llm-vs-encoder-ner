import { useEffect, useMemo, useState } from "react";
import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Alert,
  CircularProgress,
  Grid,
  MenuItem,
  Select,
  Stack,
  Typography,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { fetchExperiments, fetchExperimentErrors } from "../api/experiments";
import { fetchErrorComparison } from "../api/metrics";
import { ErrorSummaryCards } from "../components/errors/ErrorSummaryCards";
import type { ErrorAnalysisResponse, ExperimentSummary } from "../types/api";

export function ErrorAnalysisPage(): JSX.Element {
  const [experiments, setExperiments] = useState<ExperimentSummary[]>([]);
  const [selectedExperimentId, setSelectedExperimentId] = useState<string>("");
  const [analysis, setAnalysis] = useState<ErrorAnalysisResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadExperiments = async (): Promise<void> => {
      try {
        const items = await fetchExperiments();
        const eligible = items.filter((item) => item.has_predictions);
        setExperiments(eligible);
        setSelectedExperimentId(eligible[0]?.id ?? "");
      } catch (loadError) {
        setError(loadError instanceof Error ? loadError.message : "Failed to load experiment list.");
      }
    };
    void loadExperiments();
  }, []);

  useEffect(() => {
    const loadAnalysis = async (): Promise<void> => {
      if (!selectedExperimentId) {
        setAnalysis(null);
        setLoading(false);
        return;
      }
      setLoading(true);
      setError(null);
      try {
        const [current] = await Promise.all([
          fetchExperimentErrors(selectedExperimentId),
          fetchErrorComparison(),
        ]);
        setAnalysis(current);
      } catch (loadError) {
        setError(loadError instanceof Error ? loadError.message : "Failed to load error analysis.");
      } finally {
        setLoading(false);
      }
    };
    void loadAnalysis();
  }, [selectedExperimentId]);

  const chartData = useMemo(
    () =>
      Object.entries(analysis?.summary ?? {}).map(([name, value]) => ({
        name,
        value: typeof value === "number" ? value : 0,
      })),
    [analysis],
  );

  return (
    <Stack spacing={3}>
      <Stack spacing={0.5}>
        <Typography variant="h4" sx={{ fontWeight: 800 }}>
          Error Analysis
        </Typography>
        <Typography color="text.secondary">
          Inspect paradigm-specific failure modes and representative examples from stored test predictions.
        </Typography>
      </Stack>

      {error ? <Alert severity="error">{error}</Alert> : null}

      <Stack direction="row" spacing={2} alignItems="center">
        <Typography variant="subtitle1">Experiment</Typography>
        <Select
          size="small"
          value={selectedExperimentId}
          onChange={(event) => setSelectedExperimentId(event.target.value)}
          sx={{ minWidth: 280 }}
        >
          {experiments.map((experiment) => (
            <MenuItem key={experiment.id} value={experiment.id}>
              {experiment.experiment_name}
            </MenuItem>
          ))}
        </Select>
      </Stack>

      {loading ? <CircularProgress /> : null}

      {analysis ? (
        <Stack spacing={3}>
          <ErrorSummaryCards summary={analysis.summary} />
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <Typography variant="h6">Error Distribution</Typography>
            </Grid>
            <Grid item xs={12}>
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" angle={-20} textAnchor="end" interval={0} height={70} />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="value" fill="#EF6C00" radius={[8, 8, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Grid>
          </Grid>

          <Stack spacing={2}>
            {Object.entries(analysis.examples).map(([category, examples]) => (
              <Accordion key={category}>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography sx={{ fontWeight: 600 }}>{category.replace(/_/g, " ")}</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Stack spacing={2}>
                    {examples.map((example, index) => (
                      <Alert key={`${category}-${index}`} severity="info">
                        <pre style={{ margin: 0, whiteSpace: "pre-wrap" }}>{JSON.stringify(example, null, 2)}</pre>
                      </Alert>
                    ))}
                  </Stack>
                </AccordionDetails>
              </Accordion>
            ))}
          </Stack>
        </Stack>
      ) : (
        <Alert severity="info">No error analysis data available for the selected experiment.</Alert>
      )}
    </Stack>
  );
}
