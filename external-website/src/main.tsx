import React, { lazy, Suspense } from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import ReplayPage from "./ReplayPage";
import "./index.css";

const TrajectoriesPage = lazy(() => import("./TrajectoriesPage"));

const pathname = window.location.pathname;
const base = import.meta.env.BASE_URL;
const sitePath = pathname.startsWith(base) ? `/${pathname.slice(base.length)}` : pathname;
const replay = sitePath.match(/^\/replays\/pokemon\/([\w-]+)\/?$/);
const trajectories = sitePath.match(/^\/trajectories(?:\/([\w-]+))?\/?$/);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    {replay ? (
      <ReplayPage id={replay[1]} />
    ) : trajectories ? (
      <Suspense fallback={null}>
        <TrajectoriesPage initialId={trajectories[1]} />
      </Suspense>
    ) : (
      <App />
    )}
  </React.StrictMode>,
);
