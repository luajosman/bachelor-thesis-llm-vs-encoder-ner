import { Paper, Typography } from "@mui/material";

export function RawOutputPanel({ rawOutput }: { rawOutput: string | null }): JSX.Element {
  return (
    <Paper variant="outlined" sx={{ p: 2, minHeight: 220, whiteSpace: "pre-wrap", fontFamily: "monospace" }}>
      <Typography variant="subtitle2" gutterBottom>
        Raw LLM Output
      </Typography>
      {rawOutput ?? "No raw output available."}
    </Paper>
  );
}
