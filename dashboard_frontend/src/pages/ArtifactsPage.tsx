import { useEffect, useState } from "react";
import {
  Alert,
  Button,
  CircularProgress,
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

import { fetchArtifactContent, fetchArtifacts, fetchExperiments } from "../api/experiments";
import type { ArtifactContent, ArtifactSummary, ExperimentSummary } from "../types/api";

export function ArtifactsPage(): JSX.Element {
  const [experiments, setExperiments] = useState<ExperimentSummary[]>([]);
  const [selectedExperimentId, setSelectedExperimentId] = useState<string>("");
  const [artifacts, setArtifacts] = useState<ArtifactSummary[]>([]);
  const [selectedArtifact, setSelectedArtifact] = useState<string>("");
  const [artifactContent, setArtifactContent] = useState<ArtifactContent | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadExperiments = async (): Promise<void> => {
      try {
        const items = await fetchExperiments();
        const eligible = items.filter((item) => item.status !== "not_started");
        setExperiments(eligible);
        setSelectedExperimentId(eligible[0]?.id ?? "");
      } catch (loadError) {
        setError(loadError instanceof Error ? loadError.message : "Failed to load experiments.");
      }
    };
    void loadExperiments();
  }, []);

  useEffect(() => {
    const loadArtifactList = async (): Promise<void> => {
      if (!selectedExperimentId) {
        setArtifacts([]);
        setLoading(false);
        return;
      }
      setLoading(true);
      setError(null);
      try {
        const items = await fetchArtifacts(selectedExperimentId);
        setArtifacts(items);
        setSelectedArtifact(items[0]?.name ?? "");
      } catch (loadError) {
        setError(loadError instanceof Error ? loadError.message : "Failed to load artifacts.");
      } finally {
        setLoading(false);
      }
    };
    void loadArtifactList();
  }, [selectedExperimentId]);

  useEffect(() => {
    const loadContent = async (): Promise<void> => {
      if (!selectedExperimentId || !selectedArtifact) {
        setArtifactContent(null);
        return;
      }
      try {
        setArtifactContent(await fetchArtifactContent(selectedExperimentId, selectedArtifact));
      } catch (loadError) {
        setError(loadError instanceof Error ? loadError.message : "Failed to load artifact content.");
      }
    };
    void loadContent();
  }, [selectedArtifact, selectedExperimentId]);

  return (
    <Stack spacing={3}>
      <Stack spacing={0.5}>
        <Typography variant="h4" sx={{ fontWeight: 800 }}>
          Artifacts
        </Typography>
        <Typography color="text.secondary">
          Inspect stored YAML, JSON and directory artifacts produced by training, inference and comparison steps.
        </Typography>
      </Stack>

      {error ? <Alert severity="error">{error}</Alert> : null}

      <Stack direction="row" spacing={2} alignItems="center">
        <Typography variant="subtitle1">Experiment</Typography>
        <Select size="small" value={selectedExperimentId} onChange={(event) => setSelectedExperimentId(event.target.value)} sx={{ minWidth: 280 }}>
          {experiments.map((experiment) => (
            <MenuItem key={experiment.id} value={experiment.id}>
              {experiment.experiment_name}
            </MenuItem>
          ))}
        </Select>
        <Button component="a" href={`/api/download/predictions/${selectedExperimentId}`} disabled={!selectedExperimentId} variant="outlined">
          Download Predictions
        </Button>
        <Button component="a" href="/api/download/comparison-table" variant="outlined">
          Download Comparison Table
        </Button>
      </Stack>

      {loading ? <CircularProgress /> : null}

      <Grid container spacing={2}>
        <Grid item xs={12} md={4}>
          <Paper variant="outlined" sx={{ height: 560, overflow: "auto" }}>
            <List disablePadding>
              {artifacts.map((artifact) => (
                <ListItemButton key={artifact.name} selected={artifact.name === selectedArtifact} onClick={() => setSelectedArtifact(artifact.name)}>
                  <ListItemText
                    primary={artifact.name}
                    secondary={`${artifact.kind}${artifact.size_bytes ? ` · ${artifact.size_bytes} bytes` : ""}`}
                  />
                </ListItemButton>
              ))}
            </List>
          </Paper>
        </Grid>
        <Grid item xs={12} md={8}>
          <Paper variant="outlined" sx={{ p: 2, minHeight: 560 }}>
            <Typography variant="h6" gutterBottom>
              {artifactContent?.artifact.name ?? "Artifact Viewer"}
            </Typography>
            <pre style={{ margin: 0, whiteSpace: "pre-wrap", overflowX: "auto" }}>
              {artifactContent ? JSON.stringify(artifactContent.content ?? artifactContent.children, null, 2) : "Select an artifact to inspect its content."}
            </pre>
          </Paper>
        </Grid>
      </Grid>
    </Stack>
  );
}
