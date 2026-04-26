import { Grid, Card, CardContent, Typography } from "@mui/material";

export function ErrorSummaryCards({ summary }: { summary: Record<string, number | string> }): JSX.Element {
  return (
    <Grid container spacing={2}>
      {Object.entries(summary).map(([key, value]) => (
        <Grid item xs={12} sm={6} md={3} key={key}>
          <Card>
            <CardContent>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                {key.replace(/_/g, " ")}
              </Typography>
              <Typography variant="h5" sx={{ fontWeight: 700 }}>
                {String(value)}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      ))}
    </Grid>
  );
}
