import { Card, CardContent, Typography, useTheme } from "@mui/material";
import { CartesianGrid, Legend, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from "recharts";

import { regimeColors } from "../../app/theme";
import type { ComparisonRow, ExperimentRegime } from "../../types/api";

interface ScatterMetricChartProps {
  rows: ComparisonRow[];
  title: string;
  xKey: "latency_ms_mean" | "vram_peak_mb" | "total_params";
  yKey: "test_f1";
  xLabel: string;
  yLabel: string;
}

export function ScatterMetricChart({ rows, title, xKey, yKey, xLabel, yLabel }: ScatterMetricChartProps): JSX.Element {
  const theme = useTheme();
  const grouped = rows.reduce<Record<ExperimentRegime, Array<Record<string, number | string>>>>(
    (accumulator, row) => {
      const xValue = row.metrics[xKey];
      const yValue = row.metrics[yKey];
      if (xValue === null || yValue === null) {
        return accumulator;
      }
      accumulator[row.regime].push({
        x: xValue,
        y: yValue,
        name: row.experiment_name,
      });
      return accumulator;
    },
    { encoder: [], zeroshot: [], lora: [] },
  );

  return (
    <Card sx={{ height: 360 }}>
      <CardContent sx={{ height: "100%" }}>
        <Typography variant="h6" gutterBottom>
          {title}
        </Typography>
        <ResponsiveContainer width="100%" height="90%">
          <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 12 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={theme.palette.divider} />
            <XAxis dataKey="x" type="number" name={xLabel} />
            <YAxis dataKey="y" type="number" name={yLabel} domain={[0, 1]} />
            <Tooltip cursor={{ strokeDasharray: "3 3" }} />
            <Legend />
            <Scatter data={grouped.encoder} name="Encoder" fill={regimeColors.encoder} />
            <Scatter data={grouped.zeroshot} name="Zero-Shot" fill={regimeColors.zeroshot} />
            <Scatter data={grouped.lora} name="LoRA" fill={regimeColors.lora} />
          </ScatterChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
