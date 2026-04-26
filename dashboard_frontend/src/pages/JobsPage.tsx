import { useEffect, useMemo, useState } from "react";
import { Alert, Button, Grid, Stack, Typography } from "@mui/material";
import { DataGrid, type GridColDef } from "@mui/x-data-grid";

import { buildJobStreamUrl, cancelJob, fetchJobLogs, fetchJobs } from "../api/jobs";
import { LiveLogViewer } from "../components/jobs/LiveLogViewer";
import type { Job } from "../types/api";

export function JobsPage(): JSX.Element {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [selectedJobId, setSelectedJobId] = useState<string>("");
  const [log, setLog] = useState("");
  const [autoScroll, setAutoScroll] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refreshJobs = async (): Promise<void> => {
    try {
      const items = await fetchJobs();
      setJobs(items);
      if (!selectedJobId && items[0]) {
        setSelectedJobId(items[0].job_id);
      }
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Failed to load jobs.");
    }
  };

  useEffect(() => {
    void refreshJobs();
  }, []);

  useEffect(() => {
    if (!selectedJobId) {
      setLog("");
      return;
    }
    let socket: WebSocket | null = null;
    let disposed = false;

    const connect = async (): Promise<void> => {
      try {
        const payload = await fetchJobLogs(selectedJobId);
        if (!disposed) {
          setLog(payload.log);
        }
      } catch {
        // Ignore initial log fetch failure here; the websocket may still succeed.
      }

      socket = new WebSocket(buildJobStreamUrl(selectedJobId));
      socket.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data) as { log?: string };
          if (!disposed && typeof message.log === "string") {
            setLog(message.log);
            void refreshJobs();
          }
        } catch {
          // Ignore malformed websocket messages.
        }
      };
    };

    void connect();
    return () => {
      disposed = true;
      socket?.close();
    };
  }, [selectedJobId]);

  const columns = useMemo<GridColDef<Job>[]>(
    () => [
      { field: "job_id", headerName: "Job ID", flex: 1 },
      { field: "action", headerName: "Action", flex: 1 },
      { field: "status", headerName: "Status", flex: 1 },
      { field: "command_label", headerName: "Command", flex: 1.4 },
      { field: "exit_code", headerName: "Exit", width: 90 },
      {
        field: "duration",
        headerName: "Duration (s)",
        width: 120,
        valueGetter: (_value, row) => (row.duration ? row.duration.toFixed(1) : "—"),
      },
    ],
    [],
  );

  const selectedJob = jobs.find((job) => job.job_id === selectedJobId) ?? null;

  return (
    <Stack spacing={3}>
      <Stack spacing={0.5}>
        <Typography variant="h4" sx={{ fontWeight: 800 }}>
          Jobs
        </Typography>
        <Typography color="text.secondary">
          Monitor whitelisted training and evaluation jobs with live log streaming.
        </Typography>
      </Stack>

      {error ? <Alert severity="error">{error}</Alert> : null}

      <Grid container spacing={2}>
        <Grid item xs={12} lg={6}>
          <div style={{ height: 520 }}>
            <DataGrid
              rows={jobs}
              columns={columns}
              getRowId={(row) => row.job_id}
              onRowClick={(params) => setSelectedJobId(params.row.job_id)}
              pageSizeOptions={[5, 10, 25]}
            />
          </div>
        </Grid>
        <Grid item xs={12} lg={6}>
          <Stack spacing={2}>
            {selectedJob ? (
              <Button
                variant="outlined"
                color="error"
                disabled={!["queued", "running"].includes(selectedJob.status)}
                onClick={async () => {
                  await cancelJob(selectedJob.job_id);
                  await refreshJobs();
                }}
              >
                Cancel Selected Job
              </Button>
            ) : null}
            <LiveLogViewer log={log} autoScroll={autoScroll} onAutoScrollChange={setAutoScroll} />
          </Stack>
        </Grid>
      </Grid>
    </Stack>
  );
}
