import { useEffect, useState } from "react";
import {
  Alert,
  Button,
  Card,
  CardContent,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Grid,
  Stack,
  Typography,
} from "@mui/material";

import { createJob, fetchJobs } from "../api/jobs";
import type { ActionName, Job } from "../types/api";

const primaryActions: Array<{ action: ActionName; label: string; description: string }> = [
  { action: "run_all", label: "Full Pipeline", description: "Encoder, zero-shot, LoRA and comparison" },
  { action: "encoder_only", label: "Encoder Only", description: "Run DeBERTa experiments only" },
  { action: "zeroshot_only", label: "Zero-Shot Only", description: "Run Qwen zero-shot inference only" },
  { action: "lora_only", label: "LoRA Only", description: "Run fine-tuned Qwen LoRA experiments only" },
];

const secondaryActions: Array<{ action: ActionName; label: string }> = [
  { action: "deberta_base", label: "DeBERTa Base" },
  { action: "deberta_large", label: "DeBERTa Large" },
  { action: "qwen35_08b_zs", label: "Qwen 0.8B Zero-Shot" },
  { action: "qwen35_08b_lora", label: "Qwen 0.8B LoRA" },
  { action: "qwen35_4b_zs", label: "Qwen 4B Zero-Shot" },
  { action: "qwen35_4b_lora", label: "Qwen 4B LoRA" },
  { action: "qwen35_27b_zs", label: "Qwen 27B Zero-Shot" },
  { action: "qwen35_27b_lora", label: "Qwen 27B LoRA" },
  { action: "compare_only", label: "Comparison Only" },
];

const expensiveActions = new Set<ActionName>(["run_all", "qwen35_27b_zs", "qwen35_27b_lora"]);

export function RunControlPage(): JSX.Element {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [pendingAction, setPendingAction] = useState<ActionName | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refreshJobs = async (): Promise<void> => {
    try {
      setJobs(await fetchJobs());
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Failed to load jobs.");
    }
  };

  useEffect(() => {
    void refreshJobs();
  }, []);

  const startAction = async (action: ActionName): Promise<void> => {
    setError(null);
    setMessage(null);
    try {
      const job = await createJob(action);
      setMessage(`Started job ${job.job_id} for action ${action}.`);
      await refreshJobs();
    } catch (startError) {
      setError(startError instanceof Error ? startError.message : "Failed to start job.");
    }
  };

  return (
    <Stack spacing={3}>
      <Stack spacing={0.5}>
        <Typography variant="h4" sx={{ fontWeight: 800 }}>
          Run Control
        </Typography>
        <Typography color="text.secondary">
          Safe execution layer for whitelisted training, inference and comparison jobs.
        </Typography>
      </Stack>

      {message ? <Alert severity="success">{message}</Alert> : null}
      {error ? <Alert severity="error">{error}</Alert> : null}

      <Grid container spacing={2}>
        {primaryActions.map((item) => (
          <Grid item xs={12} md={6} key={item.action}>
            <Card>
              <CardContent>
                <Typography variant="h6">{item.label}</Typography>
                <Typography color="text.secondary" sx={{ mb: 2 }}>
                  {item.description}
                </Typography>
                <Button
                  variant="contained"
                  onClick={() => (expensiveActions.has(item.action) ? setPendingAction(item.action) : void startAction(item.action))}
                >
                  Start
                </Button>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      <Stack spacing={1}>
        <Typography variant="h6">Single Actions</Typography>
        <Grid container spacing={2}>
          {secondaryActions.map((item) => (
            <Grid item xs={12} sm={6} md={4} key={item.action}>
              <Button
                fullWidth
                variant="outlined"
                onClick={() => (expensiveActions.has(item.action) ? setPendingAction(item.action) : void startAction(item.action))}
              >
                {item.label}
              </Button>
            </Grid>
          ))}
        </Grid>
      </Stack>

      <Stack spacing={1}>
        <Typography variant="h6">Recent Jobs</Typography>
        {jobs.slice(0, 5).map((job) => (
          <Alert key={job.job_id} severity={job.status === "failed" ? "error" : job.status === "completed" ? "success" : "info"}>
            {job.job_id} · {job.action} · {job.status}
          </Alert>
        ))}
      </Stack>

      <Dialog open={pendingAction !== null} onClose={() => setPendingAction(null)}>
        <DialogTitle>Confirm expensive run</DialogTitle>
        <DialogContent>
          <Typography>
            {pendingAction} may require significant runtime and VRAM. Start the job anyway?
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setPendingAction(null)}>Cancel</Button>
          <Button
            variant="contained"
            onClick={() => {
              if (pendingAction) {
                void startAction(pendingAction);
              }
              setPendingAction(null);
            }}
          >
            Start
          </Button>
        </DialogActions>
      </Dialog>
    </Stack>
  );
}
