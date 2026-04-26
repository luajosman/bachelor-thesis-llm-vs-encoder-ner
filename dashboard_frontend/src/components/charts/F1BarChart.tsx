import { Card, CardContent, Typography, useTheme } from "@mui/material";
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { regimeColors } from "../../app/theme";
import type { ComparisonRow } from "../../types/api";

export function F1BarChart({ rows, title = "F1 by Model" }: { rows: ComparisonRow[]; title?: string }): JSX.Element {
  const theme = useTheme();
  const data = rows.map((row) => ({
    name: row.experiment_name,
    f1: row.metrics.test_f1 ?? 0,
    regime: row.regime,
  }));

  return (
    <Card sx={{ height: 380 }}>
      <CardContent sx={{ height: "100%" }}>
        <Typography variant="h6" gutterBottom>
          {title}
        </Typography>
        <ResponsiveContainer width="100%" height="90%">
          <BarChart data={data} margin={{ top: 12, right: 24, left: 0, bottom: 70 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={theme.palette.divider} />
            <XAxis dataKey="name" angle={-25} textAnchor="end" interval={0} height={80} />
            <YAxis domain={[0, 1]} />
            <Tooltip />
            <Bar dataKey="f1" radius={[8, 8, 0, 0]}>
              {data.map((entry) => (
                <Cell key={entry.name} fill={regimeColors[entry.regime]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
