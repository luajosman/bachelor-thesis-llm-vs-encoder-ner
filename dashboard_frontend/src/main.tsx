import React from "react";
import ReactDOM from "react-dom/client";

import App from "./app/App";
import { ThemeModeProvider } from "./app/theme";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <ThemeModeProvider>
      <App />
    </ThemeModeProvider>
  </React.StrictMode>,
);
