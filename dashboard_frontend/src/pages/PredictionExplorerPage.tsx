import { useEffect, useMemo, useState } from "react";
import {
  Alert,
  Box,
  Chip,
  CircularProgress,
  Divider,
  Grid,
  List,
  ListItemButton,
  ListItemText,
  MenuItem,
  Paper,
  Select,
  Stack,
  Typography,
} from "@mui/material";

import { fetchExperiments, fetchExperimentPredictions } from "../api/experiments";
import { RawOutputPanel } from "../components/predictions/RawOutputPanel";
import { TokenHighlighter } from "../components/predictions/TokenHighlighter";
import { RegimeChip } from "../components/metrics/RegimeChip";
import type { ExperimentSummary, PredictionSample } from "../types/api";

export function PredictionExplorerPage(): JSX.Element {
  const [experiments, setExperiments] = useState<ExperimentSummary[]>([]);
  const [selectedExperimentId, setSelectedExperimentId] = useState<string>("");
  const [predictions, setPredictions] = useState<PredictionSample[]>([]);
  const [selectedSampleId, setSelectedSampleId] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadExperiments = async (): Promise<void> => {
      try {
        const items = await fetchExperiments();
        const eligible = items.filter((item) => item.has_predictions);
        setExperiments(eligible);
        if (eligible.length > 0) {
          setSelectedExperimentId(eligible[0].id);
        }
      } catch (loadError) {
        setError(loadError instanceof Error ? loadError.message : "Failed to load experiments.");
      }
    };
    void loadExperiments();
  }, []);

  useEffect(() => {
    const loadPredictions = async (): Promise<void> => {
      if (!selectedExperimentId) {
        setPredictions([]);
        setSelectedSampleId("");
        setLoading(false);
        return;
      }
      setLoading(true);
      setError(null);
      try {
        const items = await fetchExperimentPredictions(selectedExperimentId);
        setPredictions(items);
        setSelectedSampleId(items[0]?.sample_id ?? "");
      } catch (loadError) {
        setError(loadError instanceof Error ? loadError.message : "Failed to load predictions.");
      } finally {
        setLoading(false);
      }
    };
    void loadPredictions();
  }, [selectedExperimentId]);

  const selectedExperiment = experiments.find((item) => item.id === selectedExperimentId) ?? null;
  const selectedSample = useMemo(
    () => predictions.find((item) => item.sample_id === selectedSampleId) ?? predictions[0] ?? null,
    [predictions, selectedSampleId],
  );

  return (
    <Stack spacing={3}>
      <Stack spacing={0.5}>
        <Typography variant="h4" sx={{ fontWeight: 800 }}>
          Prediction Explorer
        </Typography>
        <Typography color="text.secondary">
          Inspect token-level BIO output, generated entities and raw LLM responses for individual test samples.
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
        {selectedExperiment ? <RegimeChip regime={selectedExperiment.regime} /> : null}
      </Stack>

      {loading ? <CircularProgress /> : null}

      <Grid container spacing={2}>
        <Grid item xs={12} md={4}>
          <Paper variant="outlined" sx={{ height: 680, overflow: "auto" }}>
            <List disablePadding>
              {predictions.map((sample) => (
                <ListItemButton
                  key={sample.sample_id}
                  selected={sample.sample_id === selectedSample?.sample_id}
                  onClick={() => setSelectedSampleId(sample.sample_id)}
                >
                  <ListItemText
                    primary={`Sample ${sample.sample_id}`}
                    secondary={sample.tokens.slice(0, 8).join(" ")}
                  />
                  {sample.parse_status ? (
                    <Chip label={sample.parse_status} size="small" color={sample.parse_status === "failed" ? "error" : "default"} />
                  ) : null}
                </ListItemButton>
              ))}
            </List>
          </Paper>
        </Grid>

        <Grid item xs={12} md={8}>
          {selectedSample ? (
            <Stack spacing={2}>
              <Paper variant="outlined" sx={{ p: 2 }}>
                <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 2 }}>
                  <Typography variant="h6">Sample {selectedSample.sample_id}</Typography>
                  {selectedSample.parse_status ? (
                    <Chip label={selectedSample.parse_status} color={selectedSample.parse_status === "failed" ? "error" : "success"} size="small" />
                  ) : null}
                </Stack>
                <TokenHighlighter title="Gold BIO" tokens={selectedSample.tokens} tags={selectedSample.gold_bio} />
                <Divider sx={{ my: 2 }} />
                <TokenHighlighter title="Predicted BIO" tokens={selectedSample.tokens} tags={selectedSample.pred_bio} />
              </Paper>

              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <Paper variant="outlined" sx={{ p: 2, minHeight: 180 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      Gold Entities
                    </Typography>
                    <Box component="pre" sx={{ m: 0, overflowX: "auto" }}>
                      {JSON.stringify(selectedSample.gold_entities, null, 2)}
                    </Box>
                  </Paper>
                </Grid>
                <Grid item xs={12} md={6}>
                  <Paper variant="outlined" sx={{ p: 2, minHeight: 180 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      Predicted Entities
                    </Typography>
                    <Box component="pre" sx={{ m: 0, overflowX: "auto" }}>
                      {JSON.stringify(selectedSample.pred_entities, null, 2)}
                    </Box>
                  </Paper>
                </Grid>
              </Grid>

              <RawOutputPanel rawOutput={selectedSample.raw_output} />
            </Stack>
          ) : (
            <Alert severity="info">No prediction samples available for the selected experiment.</Alert>
          )}
        </Grid>
      </Grid>
    </Stack>
  );
}
