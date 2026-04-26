import { Card, CardContent, Typography, useTheme } from "@mui/material";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import type { ComparisonRow } from "../../types/api";

export function PrecisionRecallChart({ rows }: { rows: ComparisonRow[] }): JSX.Element {
  const theme = useTheme();
  const data = rows.map((row) => ({
    name: row.experiment_name,
    precision: row.metrics.test_precision ?? 0,
    recall: row.metrics.test_recall ?? 0,
  }));

  return (
    <Card sx={{ height: 380 }}>
      <CardContent sx={{ height: "100%" }}>
        <Typography variant="h6" gutterBottom>
          Precision vs Recall
        </Typography>
        <ResponsiveContainer width="100%" height="90%">
          <BarChart data={data} margin={{ top: 12, right: 24, left: 0, bottom: 70 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={theme.palette.divider} />
            <XAxis dataKey="name" angle={-25} textAnchor="end" interval={0} height={80} />
            <YAxis domain={[0, 1]} />
            <Tooltip />
            <Legend />
            <Bar dataKey="precision" fill="#4FC3F7" radius={[8, 8, 0, 0]} />
            <Bar dataKey="recall" fill="#81C784" radius={[8, 8, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
