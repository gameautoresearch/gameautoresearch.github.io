import { useSyncExternalStore } from "react";
import "./view.css";

// Agent view: a plain-text rendering of every page for LLMs and text browsers. The choice lives in a tiny module
// store (no provider needed), is remembered in localStorage, and can be set by link with ?view=agent or ?view=human.
// It puts the class "agent-view" on <html>, which view.css keys off; components that can't be flattened with CSS
// alone (charts, the replay, asides) read useAgentView() and render a text variant.

const STORAGE_KEY = "view";
const listeners = new Set<() => void>();

function initial() {
  const param = new URLSearchParams(window.location.search).get("view");
  if (param === "agent" || param === "human") {
    save(param === "agent");
    return param === "agent";
  }
  try {
    return window.localStorage.getItem(STORAGE_KEY) === "agent";
  } catch {
    return false;
  }
}

function save(agent: boolean) {
  try {
    window.localStorage.setItem(STORAGE_KEY, agent ? "agent" : "human");
  } catch {
    // Storage blocked: the choice still holds for this visit.
  }
}

let agentView = initial();
document.documentElement.classList.toggle("agent-view", agentView);

export function setAgentView(agent: boolean) {
  if (agent === agentView) return;
  agentView = agent;
  save(agent);
  document.documentElement.classList.toggle("agent-view", agent);
  // Keep the address shareable: agent view carries ?view=agent, human view drops the parameter.
  const url = new URL(window.location.href);
  if (agent) url.searchParams.set("view", "agent");
  else url.searchParams.delete("view");
  window.history.replaceState(window.history.state, "", url);
  listeners.forEach((notify) => notify());
}

const subscribe = (notify: () => void) => {
  listeners.add(notify);
  return () => listeners.delete(notify);
};

export const useAgentView = () => useSyncExternalStore(subscribe, () => agentView);

export function ViewToggle() {
  const agent = useAgentView();
  return (
    <button type="button" className="view-toggle" onClick={() => setAgentView(!agent)}>
      {agent ? "Human view" : "Agent view"}
    </button>
  );
}
