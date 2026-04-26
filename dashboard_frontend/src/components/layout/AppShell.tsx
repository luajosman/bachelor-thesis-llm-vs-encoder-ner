import AnalyticsIcon from "@mui/icons-material/Analytics";
import CompareArrowsIcon from "@mui/icons-material/CompareArrows";
import DescriptionIcon from "@mui/icons-material/Description";
import ErrorOutlineIcon from "@mui/icons-material/ErrorOutline";
import InsightsIcon from "@mui/icons-material/Insights";
import ListAltIcon from "@mui/icons-material/ListAlt";
import MenuIcon from "@mui/icons-material/Menu";
import PlayCircleOutlineIcon from "@mui/icons-material/PlayCircleOutline";
import TableChartIcon from "@mui/icons-material/TableChart";
import {
  AppBar,
  Box,
  Divider,
  Drawer,
  IconButton,
  List,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Toolbar,
  Tooltip,
  Typography,
  useMediaQuery,
  useTheme,
} from "@mui/material";
import { useState } from "react";
import { NavLink, Outlet } from "react-router-dom";

import { ThemeToggle } from "./ThemeToggle";

const expandedDrawerWidth = 260;
const collapsedDrawerWidth = 88;

const navigation = [
  { label: "Overview", path: "/", icon: <AnalyticsIcon /> },
  { label: "Experiments", path: "/experiments", icon: <TableChartIcon /> },
  { label: "Compare", path: "/compare", icon: <CompareArrowsIcon /> },
  { label: "Predictions", path: "/predictions", icon: <InsightsIcon /> },
  { label: "Error Analysis", path: "/errors", icon: <ErrorOutlineIcon /> },
  { label: "Run Control", path: "/run-control", icon: <PlayCircleOutlineIcon /> },
  { label: "Jobs", path: "/jobs", icon: <ListAltIcon /> },
  { label: "Artifacts", path: "/artifacts", icon: <DescriptionIcon /> },
];

export function AppShell(): JSX.Element {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down("md"));
  const [mobileOpen, setMobileOpen] = useState(false);
  const [desktopCollapsed, setDesktopCollapsed] = useState(false);

  const currentDrawerWidth = isMobile
    ? expandedDrawerWidth
    : desktopCollapsed
      ? collapsedDrawerWidth
      : expandedDrawerWidth;

  const handleDrawerToggle = (): void => {
    if (isMobile) {
      setMobileOpen((previous) => !previous);
      return;
    }
    setDesktopCollapsed((previous) => !previous);
  };

  const closeMobileDrawer = (): void => {
    if (isMobile) {
      setMobileOpen(false);
    }
  };

  const drawerContent = (
    <>
      <Toolbar
        sx={{
          px: 2,
          display: "flex",
          alignItems: "center",
          justifyContent: desktopCollapsed && !isMobile ? "center" : "flex-start",
        }}
      >
        {!desktopCollapsed || isMobile ? (
          <Typography variant="h6" sx={{ fontWeight: 800 }}>
            Dashboard
          </Typography>
        ) : (
          <Typography variant="subtitle1" sx={{ fontWeight: 800 }}>
            D
          </Typography>
        )}
      </Toolbar>
      <Divider />
      <List sx={{ px: 1, py: 1 }}>
        {navigation.map((item) => {
          const button = (
            <ListItemButton
              key={item.path}
              component={NavLink}
              to={item.path}
              onClick={closeMobileDrawer}
              sx={{
                mb: 0.5,
                minHeight: 52,
                borderRadius: 3,
                justifyContent: desktopCollapsed && !isMobile ? "center" : "flex-start",
                px: desktopCollapsed && !isMobile ? 1.5 : 2,
                "&.active": {
                  bgcolor: "action.selected",
                },
              }}
            >
              <ListItemIcon
                sx={{
                  minWidth: desktopCollapsed && !isMobile ? 0 : 42,
                  justifyContent: "center",
                  color: "inherit",
                }}
              >
                {item.icon}
              </ListItemIcon>
              {!desktopCollapsed || isMobile ? <ListItemText primary={item.label} /> : null}
            </ListItemButton>
          );

          if (desktopCollapsed && !isMobile) {
            return (
              <Tooltip key={item.path} title={item.label} placement="right">
                {button}
              </Tooltip>
            );
          }

          return button;
        })}
      </List>
    </>
  );

  return (
    <Box sx={{ display: "flex", minHeight: "100vh" }}>
      <AppBar
        position="fixed"
        color="inherit"
        elevation={0}
        sx={{
          width: isMobile ? "100%" : `calc(100% - ${currentDrawerWidth}px)`,
          ml: isMobile ? 0 : `${currentDrawerWidth}px`,
          borderBottom: "1px solid",
          borderColor: "divider",
          transition: theme.transitions.create(["width", "margin"], {
            duration: theme.transitions.duration.shorter,
          }),
        }}
      >
        <Toolbar>
          <IconButton color="inherit" edge="start" onClick={handleDrawerToggle} sx={{ mr: 1.5 }}>
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" sx={{ flexGrow: 1, fontWeight: 700 }}>
            BA-NER Experiment Dashboard
          </Typography>
          <ThemeToggle />
        </Toolbar>
      </AppBar>

      {isMobile ? (
        <Drawer
          variant="temporary"
          open={mobileOpen}
          onClose={closeMobileDrawer}
          ModalProps={{ keepMounted: true }}
          sx={{
            display: { xs: "block", md: "none" },
            [`& .MuiDrawer-paper`]: {
              width: expandedDrawerWidth,
              boxSizing: "border-box",
              borderRight: "1px solid",
              borderColor: "divider",
            },
          }}
        >
          {drawerContent}
        </Drawer>
      ) : (
        <Drawer
          variant="permanent"
          sx={{
            width: currentDrawerWidth,
            flexShrink: 0,
            [`& .MuiDrawer-paper`]: {
              width: currentDrawerWidth,
              overflowX: "hidden",
              boxSizing: "border-box",
              borderRight: "1px solid",
              borderColor: "divider",
              transition: theme.transitions.create("width", {
                duration: theme.transitions.duration.shorter,
              }),
            },
          }}
        >
          {drawerContent}
        </Drawer>
      )}

      <Box
        component="main"
        sx={{
          flexGrow: 1,
          minWidth: 0,
          px: { xs: 2, sm: 3 },
          py: { xs: 2, sm: 3 },
          mt: 8,
        }}
      >
        <Outlet />
      </Box>
    </Box>
  );
}
