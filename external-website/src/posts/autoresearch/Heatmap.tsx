import { Fragment } from "react";
import { liftFill } from "./colors";
import { useAgentView } from "../../view";

// Pokémon 3v3, mean score of the row model's networks against the column model's (reports/pokemon/poke3v3-rep-2026-11-15.md).
const NAMES = ["Sol", "Astra", "Luna", "Inkling", "DeepSeek", "Kimi", "GLM", "Starter"];
const M: (number | null)[][] = [
  [null, 50.8, 59.8, 60.2, 61.7, 60.9, 65.6, 68.2],
  [49.2, null, 59.0, 59.4, 60.7, 59.4, 63.6, 65.5],
  [40.2, 41.0, null, 51.5, 52.6, 51.0, 55.5, 58.2],
  [39.8, 40.6, 48.5, null, 51.3, 51.1, 55.2, 56.7],
  [38.3, 39.3, 47.4, 48.7, null, 49.1, 53.7, 54.3],
  [39.1, 40.6, 49.0, 48.9, 50.9, null, 53.5, 53.7],
  [34.4, 36.4, 44.5, 44.8, 46.3, 46.5, null, 51.7],
  [31.8, 34.5, 41.8, 43.3, 45.7, 46.3, 48.3, null],
];

function color(v: number) {
  return liftFill(Math.max(-1, Math.min(1, (v - 50) / 18)));
}

export default function Heatmap() {
  const agent = useAgentView();
  if (agent) {
    return (
      <div className="agent-table">
        <table>
          <thead><tr><th>Row vs column</th>{NAMES.map((n) => <th key={n}>{n}</th>)}</tr></thead>
          <tbody>{M.map((r, i) => <tr key={i}><td>{NAMES[i]}</td>{r.map((v, j) => <td key={j}>{v ?? "–"}</td>)}</tr>)}</tbody>
        </table>
      </div>
    );
  }
  return (
    <div className="ar-heat" style={{ gridTemplateColumns: `76px repeat(${NAMES.length}, 1fr)` }}>
      <span />
      {NAMES.map((n) => <span key={n} className="ar-heat-col">{n}</span>)}
      {M.map((r, i) => (
        <Fragment key={i}>
          <span className="ar-heat-row">{NAMES[i]}</span>
          {r.map((v, j) => (
            <span key={`${i}-${j}`} className="ar-heat-cell" title={v === null ? "" : `${NAMES[i]} vs ${NAMES[j]}: ${v}%`}
              style={{ background: v === null ? "transparent" : color(v), color: v !== null && Math.abs(v - 50) > 11 ? "#fff" : "var(--ink)" }}>
              {v === null ? "" : v.toFixed(0)}
            </span>
          ))}
        </Fragment>
      ))}
    </div>
  );
}
