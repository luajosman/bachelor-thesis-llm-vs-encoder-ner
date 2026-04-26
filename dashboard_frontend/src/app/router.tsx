import { createBrowserRouter } from "react-router-dom";

import { AppShell } from "../components/layout/AppShell";
import { ArtifactsPage } from "../pages/ArtifactsPage";
import { ComparePage } from "../pages/ComparePage";
import { ErrorAnalysisPage } from "../pages/ErrorAnalysisPage";
import { ExperimentsPage } from "../pages/ExperimentsPage";
import { JobsPage } from "../pages/JobsPage";
import { OverviewPage } from "../pages/OverviewPage";
import { PredictionExplorerPage } from "../pages/PredictionExplorerPage";
import { RunControlPage } from "../pages/RunControlPage";

export const router = createBrowserRouter([
  {
    path: "/",
    element: <AppShell />,
    children: [
      { index: true, element: <OverviewPage /> },
      { path: "experiments", element: <ExperimentsPage /> },
      { path: "compare", element: <ComparePage /> },
      { path: "predictions", element: <PredictionExplorerPage /> },
      { path: "errors", element: <ErrorAnalysisPage /> },
      { path: "run-control", element: <RunControlPage /> },
      { path: "jobs", element: <JobsPage /> },
      { path: "artifacts", element: <ArtifactsPage /> },
    ],
  },
]);
