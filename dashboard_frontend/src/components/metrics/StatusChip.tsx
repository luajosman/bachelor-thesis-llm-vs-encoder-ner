import { Chip } from "@mui/material";

import type { ExperimentStatus } from "../../types/api";
import { regimeColors } from "../../app/theme";

const statusColors: Record<ExperimentStatus, string> = {
  not_started: regimeColors.missing,
  trained: "#0288D1",
  inferred: "#6D4C41",
  complete: regimeColors.encoder,
  failed: regimeColors.failed,
};

export function StatusChip({ status }: { status: ExperimentStatus }): JSX.Element {
  return (
    <Chip
      label={status.replace("_", " ")}
      size="small"
      sx={{
        textTransform: "capitalize",
        bgcolor: `${statusColors[status]}18`,
        color: statusColors[status],
        fontWeight: 600,
      }}
    />
  );
}
