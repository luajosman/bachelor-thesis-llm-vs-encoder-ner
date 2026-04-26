import { Card, CardContent, Typography } from "@mui/material";
import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";

import { regimeColors } from "../../app/theme";
import type { ComparisonRow } from "../../types/api";

export function RegimeDistributionChart({ rows }: { rows: ComparisonRow[] }): JSX.Element {
  const counts = rows.reduce<Record<string, number>>((accumulator, row) => {
    accumulator[row.regime] = (accumulator[row.regime] ?? 0) + 1;
    return accumulator;
  }, {});
  const data = [
    { name: "Encoder", value: counts.encoder ?? 0, color: regimeColors.encoder },
    { name: "Zero-Shot", value: counts.zeroshot ?? 0, color: regimeColors.zeroshot },
    { name: "LoRA", value: counts.lora ?? 0, color: regimeColors.lora },
  ].filter((entry) => entry.value > 0);

  return (
    <Card sx={{ height: 380 }}>
      <CardContent sx={{ height: "100%" }}>
        <Typography variant="h6" gutterBottom>
          Regime Distribution
        </Typography>
        <ResponsiveContainer width="100%" height="90%">
          <PieChart>
            <Pie data={data} dataKey="value" innerRadius={72} outerRadius={110} paddingAngle={3}>
              {data.map((entry) => (
                <Cell key={entry.name} fill={entry.color} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
