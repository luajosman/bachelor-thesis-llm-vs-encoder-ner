import { Card, CardContent, Stack, Typography, useTheme } from "@mui/material";

import type { PerEntityComparisonResponse } from "../../types/api";

function getCellColor(value: number, darkMode: boolean): string {
  const alpha = Math.max(0.1, Math.min(0.95, value));
  return darkMode ? `rgba(76, 175, 80, ${alpha})` : `rgba(46, 125, 50, ${alpha})`;
}

export function Heatmap({ data }: { data: PerEntityComparisonResponse }): JSX.Element {
  const theme = useTheme();
  const isDark = theme.palette.mode === "dark";

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Per-Entity F1 Heatmap
        </Typography>
        <Stack spacing={1} sx={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr>
                <th style={{ textAlign: "left", padding: 8 }}>Experiment</th>
                {data.entity_types.map((entity) => (
                  <th key={entity} style={{ padding: 8, textAlign: "center" }}>
                    {entity}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.experiments.map((experiment) => (
                <tr key={experiment.experiment_id}>
                  <td style={{ padding: 8, whiteSpace: "nowrap", fontWeight: 600 }}>{experiment.experiment_name}</td>
                  {data.entity_types.map((entity) => {
                    const value = experiment.metrics[entity]?.f1 ?? 0;
                    return (
                      <td
                        key={`${experiment.experiment_id}-${entity}`}
                        style={{
                          padding: 8,
                          textAlign: "center",
                          backgroundColor: getCellColor(value, isDark),
                          color: value > 0.65 ? "#fff" : theme.palette.text.primary,
                          borderRadius: 8,
                        }}
                      >
                        {value.toFixed(2)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </Stack>
      </CardContent>
    </Card>
  );
}
