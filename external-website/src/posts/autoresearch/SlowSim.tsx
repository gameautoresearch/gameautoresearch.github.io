// The slow-simulator Generals run (readout reports/generals/slowsim-2026-11-20.md in the bench repository).
// Training speed: median joint ticks (one tick of one game) per second of training, between consecutive progress rows
// of a run; Opus logged seat-samples, converted from its config (512 games x 64 ticks per 2.25-s iteration).
// Tournament: every pair of the four submitted networks on 128 private maps, both seats; capture win 1, draw 0.5.
import { useAgentView } from "../../view";
import { modelStyle } from "./colors";

export const SLOW = [
  { model: "Claude Opus 5.5", rebuilt: true, minute: 1.2, speed: 14600, score: 91.4,
    did: "Rebuilt it as a batched simulator at minute 1, checked it against the original" },
  { model: "GPT-6.1 Sol", rebuilt: true, minute: 1.7, speed: 5400, score: 70.8,
    did: "Rebuilt it at minute 2, re-checks it against the original before every run" },
  { model: "DeepSeek V4 Pro", rebuilt: false, minute: null, speed: 565, score: 22.5,
    did: "Timed it at minute 37 and called it very slow; kept it" },
  { model: "Kimi K3.1 Pro", rebuilt: false, minute: null, speed: 630, score: 15.3,
    did: "Called it the bottleneck at minute 1; kept it" },
];

const fmt = (v: number) => v.toLocaleString("en-US");
// Log scale for speed, 100 to 20,000 joint ticks a second.
const speedPct = (v: number) => ((Math.log10(v) - 2) / (Math.log10(20000) - 2)) * 100;

export function SlowSimBoard() {
  if (useAgentView()) {
    return (
      <ol>
        {SLOW.map((x) => (
          <li key={x.model}>
            {x.model}: {x.did}. Trained at about {fmt(x.speed)} game ticks a second; tournament score {x.score}.
          </li>
        ))}
      </ol>
    );
  }
  return (
    <div className="fr-board ss-board">
      <table>
        <thead>
          <tr>
            <th>Researcher</th>
            <th>Training speed (game ticks/s)</th>
            <th>Tournament (0–100)</th>
          </tr>
        </thead>
        <tbody>
          {SLOW.map((x) => {
            const st = modelStyle(x.model);
            return (
              <tr key={x.model}>
                <td>
                  <span className="fr-name"><i style={{ background: st.color }} />{x.model}</span>
                  <small>{x.did}</small>
                </td>
                <td>
                  <span className="fr-num">{fmt(x.speed)}</span>
                  <span className="fr-bar"><b style={{ left: 0, width: `${speedPct(x.speed)}%`, background: st.color }} /></span>
                </td>
                <td className="fr-on">
                  <span className="fr-num">{x.score.toFixed(1)}</span>
                  <span className="fr-bar"><b style={{ left: 0, width: `${x.score}%`, background: st.color }} /></span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
