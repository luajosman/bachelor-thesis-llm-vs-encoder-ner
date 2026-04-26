import { Paper, Stack, Switch, Typography } from "@mui/material";
import { useEffect, useRef } from "react";

interface LiveLogViewerProps {
  log: string;
  autoScroll: boolean;
  onAutoScrollChange: (value: boolean) => void;
}

export function LiveLogViewer({ log, autoScroll, onAutoScrollChange }: LiveLogViewerProps): JSX.Element {
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (autoScroll && ref.current) {
      ref.current.scrollTop = ref.current.scrollHeight;
    }
  }, [autoScroll, log]);

  return (
    <Stack spacing={1}>
      <Stack direction="row" alignItems="center" justifyContent="space-between">
        <Typography variant="h6">Live Logs</Typography>
        <Stack direction="row" spacing={1} alignItems="center">
          <Typography variant="body2">Auto-scroll</Typography>
          <Switch checked={autoScroll} onChange={(_, checked) => onAutoScrollChange(checked)} />
        </Stack>
      </Stack>
      <Paper
        ref={ref}
        variant="outlined"
        sx={{
          p: 2,
          minHeight: 420,
          maxHeight: 420,
          overflow: "auto",
          fontFamily: "monospace",
          whiteSpace: "pre-wrap",
        }}
      >
        {log || "No logs yet."}
      </Paper>
    </Stack>
  );
}
