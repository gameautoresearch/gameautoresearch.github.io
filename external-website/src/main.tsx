import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import ReplayPage from "./ReplayPage";
import "./index.css";

// The site is one post; the only other page is the full-size view of a recorded Pokémon battle.
const pathname = window.location.pathname;
const base = import.meta.env.BASE_URL;
const sitePath = pathname.startsWith(base) ? `/${pathname.slice(base.length)}` : pathname;
const replay = sitePath.match(/^\/replays\/pokemon\/([\w-]+)\/?$/);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    {replay ? <ReplayPage id={replay[1]} /> : <App />}
  </React.StrictMode>,
);
