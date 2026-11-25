import React from "react";
import ReactDOM from "react-dom/client";
import "../index.css";
import "../posts/shared/post.css";
import "../posts/autoresearch/post.css";
import "./video.css";
import TwitterVideo from "./TwitterVideo";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <TwitterVideo />
  </React.StrictMode>,
);
