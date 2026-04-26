import { Box } from "@mui/material";
import { DataGrid, GridToolbar, type GridColDef } from "@mui/x-data-grid";

import { formatMegabytes, formatMetric, formatMilliseconds, formatParams } from "../../app/utils";
import type { ComparisonRow } from "../../types/api";
import { RegimeChip } from "../metrics/RegimeChip";
import { StatusChip } from "../metrics/StatusChip";

export function ExperimentTable({ rows }: { rows: ComparisonRow[] }): JSX.Element {
  const columns: GridColDef<ComparisonRow>[] = [
    { field: "experiment_name", headerName: "Experiment", flex: 1.4, minWidth: 180 },
    { field: "model_name", headerName: "Model", flex: 1.2, minWidth: 180 },
    {
      field: "regime",
      headerName: "Regime",
      minWidth: 120,
      renderCell: (params) => <RegimeChip regime={params.row.regime} />,
    },
    { field: "dataset", headerName: "Dataset", minWidth: 110 },
    {
      field: "status",
      headerName: "Status",
      minWidth: 120,
      renderCell: (params) => <StatusChip status={params.row.status} />,
    },
    { field: "test_f1", headerName: "F1", minWidth: 90, valueGetter: (_value, row) => formatMetric(row.metrics.test_f1) },
    { field: "test_precision", headerName: "Precision", minWidth: 90, valueGetter: (_value, row) => formatMetric(row.metrics.test_precision) },
    { field: "test_recall", headerName: "Recall", minWidth: 90, valueGetter: (_value, row) => formatMetric(row.metrics.test_recall) },
    { field: "latency", headerName: "Latency", minWidth: 110, valueGetter: (_value, row) => formatMilliseconds(row.metrics.latency_ms_mean) },
    { field: "vram", headerName: "VRAM", minWidth: 110, valueGetter: (_value, row) => formatMegabytes(row.metrics.vram_peak_mb) },
    { field: "params", headerName: "Params", minWidth: 100, valueGetter: (_value, row) => formatParams(row.metrics.total_params) },
  ];

  return (
    <Box sx={{ height: 640 }}>
      <DataGrid
        rows={rows}
        columns={columns}
        getRowId={(row) => row.experiment_id}
        disableRowSelectionOnClick
        slots={{ toolbar: GridToolbar }}
        pageSizeOptions={[10, 25, 50]}
        initialState={{
          pagination: { paginationModel: { pageSize: 10, page: 0 } },
        }}
      />
    </Box>
  );
}
