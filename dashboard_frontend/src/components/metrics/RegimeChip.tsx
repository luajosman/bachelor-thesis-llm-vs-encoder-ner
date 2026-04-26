import { Chip } from "@mui/material";

import type { ExperimentRegime } from "../../types/api";
import { regimeColors } from "../../app/theme";

const labels: Record<ExperimentRegime, string> = {
  encoder: "Encoder",
  zeroshot: "Zero-Shot",
  lora: "LoRA",
};

export function RegimeChip({ regime }: { regime: ExperimentRegime }): JSX.Element {
  return (
    <Chip
      label={labels[regime]}
      size="small"
      sx={{
        bgcolor: `${regimeColors[regime]}18`,
        color: regimeColors[regime],
        fontWeight: 600,
      }}
    />
  );
}
