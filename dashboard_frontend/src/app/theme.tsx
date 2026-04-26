import { createContext, useContext, useMemo, useState, type PropsWithChildren } from "react";
import { CssBaseline, ThemeProvider, createTheme } from "@mui/material";

export const regimeColors = {
  encoder: "#2E7D32",
  zeroshot: "#1976D2",
  lora: "#7B1FA2",
  failed: "#D32F2F",
  missing: "#ED6C02",
};

type ThemeMode = "light" | "dark";

interface ThemeModeContextValue {
  mode: ThemeMode;
  toggleMode: () => void;
}

const ThemeModeContext = createContext<ThemeModeContextValue | undefined>(undefined);

export function ThemeModeProvider({ children }: PropsWithChildren): JSX.Element {
  const [mode, setMode] = useState<ThemeMode>(() => {
    const stored = window.localStorage.getItem("dashboard-theme-mode");
    return stored === "dark" ? "dark" : "light";
  });

  const toggleMode = (): void => {
    setMode((previous) => {
      const next = previous === "light" ? "dark" : "light";
      window.localStorage.setItem("dashboard-theme-mode", next);
      return next;
    });
  };

  const theme = useMemo(
    () =>
      createTheme({
        palette: {
          mode,
          primary: { main: "#3B82F6" },
          secondary: { main: regimeColors.lora },
          background:
            mode === "light"
              ? { default: "#F5F7FB", paper: "#FFFFFF" }
              : { default: "#0F172A", paper: "#111827" },
        },
        shape: {
          borderRadius: 14,
        },
        components: {
          MuiCard: {
            styleOverrides: {
              root: {
                boxShadow: "0 8px 28px rgba(15, 23, 42, 0.08)",
                border: "1px solid rgba(148, 163, 184, 0.18)",
              },
            },
          },
          MuiPaper: {
            styleOverrides: {
              root: {
                backgroundImage: "none",
              },
            },
          },
        },
      }),
    [mode],
  );

  return (
    <ThemeModeContext.Provider value={{ mode, toggleMode }}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        {children}
      </ThemeProvider>
    </ThemeModeContext.Provider>
  );
}

export function useThemeMode(): ThemeModeContextValue {
  const context = useContext(ThemeModeContext);
  if (!context) {
    throw new Error("useThemeMode must be used inside ThemeModeProvider");
  }
  return context;
}
