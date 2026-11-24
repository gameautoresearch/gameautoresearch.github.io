import { useAgentView } from "../../view";
import { Table } from "../shared/text";

// Gemini 4 Neon's $8.70 Polytopia recipe session (`polytopia-neon-b3`, $8.58 spent).
// Sourced from its research ledger (`ledger.sqlite3`) and `NOTES.md`.
// Each branch's lift is its 512-game practice win-rate change over the best prior 12-minute result:
// base0 0.00% -> v1b 3.13% (+3.1%) -> v3a 11.33% (+8.2%) / v3b 15.23% (+12.1%)
// -> v4a 18.36% (+3.1%) / v4b 15.43% (+0.2%) -> v5a 22.66% (+4.3%) / v5b 21.68% (+3.3%) -> v6final 47.85%.

type Branch = {
  label: string;
  delta?: { dir: "up" | "down"; text: string };
  status?: string;
  crash?: string;
};

type Row = {
  key: string;
  costUsd: number;
  cpuUsd?: number;
  cost: string;
  time: string;
  a: Branch;
  b: Branch;
  winner?: "a" | "b";
  best: string;
};

const BUDGET = 8.7;
const SPENT = 8.58;
const UNUSED = BUDGET - SPENT;
const PX_PER_USD = 46;

const ROWS: Row[] = [
  {
    key: "r1",
    costUsd: 0.9,
    cost: "$0.90",
    time: "5m + 12m",
    a: { label: "Starter recipe", status: "baseline (0.0%)" },
    b: {
      label: "Dual-view critic + turn credit",
      delta: { dir: "up", text: "+3.1%" },
    },
    winner: "b",
    best: "3.1%",
  },
  {
    key: "r2",
    costUsd: 1.23,
    cpuUsd: 0.04,
    cost: "$1.27",
    time: "2×12m + CPU",
    a: { label: "Pyramid net, self-play", crash: "Eval crash (discarded)" },
    b: {
      label: "Pyramid net + opponent pool",
      crash: "Eval crash · fixed on CPU ($0.04)",
    },
    best: "—",
  },
  {
    key: "r3",
    costUsd: 1.22,
    cost: "$1.22",
    time: "2×12m",
    a: {
      label: "Fixed pyramid net, self-play",
      delta: { dir: "up", text: "+8.2%" },
    },
    b: {
      label: "Fixed pyramid + opponent pool",
      delta: { dir: "up", text: "+12.1%" },
    },
    winner: "b",
    best: "15.2%",
  },
  {
    key: "r4",
    costUsd: 1.23,
    cost: "$1.23",
    time: "2×12m",
    a: {
      label: "1-epoch, standard critic",
      delta: { dir: "up", text: "+3.1%" },
    },
    b: {
      label: "2-epoch, distributional critic",
      delta: { dir: "up", text: "+0.2%" },
    },
    winner: "a",
    best: "18.4%",
  },
  {
    key: "r5",
    costUsd: 1.23,
    cost: "$1.23",
    time: "2×12m",
    a: {
      label: "Smaller batches (more updates)",
      delta: { dir: "up", text: "+4.3%" },
    },
    b: {
      label: "Larger batches + extra shaping",
      delta: { dir: "up", text: "+3.3%" },
    },
    winner: "a",
    best: "22.7%",
  },
];

const LEGEND = [
  { key: "ab", label: "5 A/B tests ($5.81)", color: "#4a71d2" },
  { key: "dbg", label: "CPU debug ($0.04)", color: "#d4a000" },
  { key: "val", label: "1 h validation ($2.73)", color: "#19337d" },
  { key: "left", label: "Unused ($0.12)", color: "#dcdbd5" },
];

function fmtBranch(br: Branch) {
  if (br.delta)
    return `${br.label} (${br.delta.dir === "up" ? "▲" : "▼"} ${br.delta.text})`;
  if (br.crash) return `${br.label} (${br.crash})`;
  return br.status ? `${br.label} (${br.status})` : br.label;
}

function BranchCell({ br, win }: { br: Branch; win: boolean }) {
  return (
    <div
      className={`ar-nd-cell ${win ? "ar-nd-win" : "ar-nd-muted"}`}
      role="cell"
    >
      <span className="ar-nd-bname">{br.label}</span>
      {br.delta && (
        <span className={`ar-nd-lift ${br.delta.dir}${win ? " is-win" : ""}`}>
          {br.delta.dir === "up" ? "▲" : "▼"} {br.delta.text}
        </span>
      )}
      {br.crash && <span className="ar-nd-lift crash">{br.crash}</span>}
      {br.status && <span className="ar-nd-sub">{br.status}</span>}
    </div>
  );
}

export default function NeonDiscipline() {
  if (useAgentView()) {
    return (
      <Table
        head={[
          "Cost & time",
          "Branch A (vs. prior best)",
          "Branch B (vs. prior best)",
          "Lift",
        ]}
        align={["l", "l", "l", "r"]}
        rows={[
          ...ROWS.map((r) => [
            `${r.cost} (${r.time})`,
            r.winner === "a" ? (
              <strong>{fmtBranch(r.a)}</strong>
            ) : (
              fmtBranch(r.a)
            ),
            r.winner === "b" ? (
              <strong>{fmtBranch(r.b)}</strong>
            ) : (
              fmtBranch(r.b)
            ),
            r.best === "—" ? "—" : `+${r.best}`,
          ]),
          [
            "$2.73 (56m)",
            "1-hour validation of winning recipe: dual-view critic and turn-based credit, spatial pyramid, scoreboard inputs, historical opponent pool, and fast 1-epoch updates on smaller batches (final submission).",
            "—",
            <strong>+47.9%</strong>,
          ],
        ]}
      />
    );
  }

  const h = (usd: number) => Math.round(usd * PX_PER_USD);

  return (
    <div className="ar-nd">
      <div className="pt-table-wrap ar-nd-wrap">
        <div
          className="ar-nd-grid"
          role="table"
          aria-label="Gemini 4 Neon's $8.70 Polytopia research budget and five A/B tests"
        >
          <div className="ar-nd-row ar-nd-head" role="row">
            <div
              className="ar-nd-bar-col"
              role="columnheader"
              aria-label="Budget share"
            />
            <div role="columnheader">Cost &amp; time</div>
            <div role="columnheader">Branch A</div>
            <div role="columnheader">Branch B</div>
            <div className="r" role="columnheader">
              Lift
            </div>
          </div>

          {ROWS.map((r, i) => {
            const totalUsd = r.costUsd + (r.cpuUsd ?? 0);
            return (
              <div
                key={r.key}
                className="ar-nd-row"
                role="row"
                style={{ minHeight: h(totalUsd) }}
              >
                <div className="ar-nd-bar-col" aria-hidden="true">
                  <span
                    className={`ar-nd-vseg${i === 0 ? " is-first" : ""}`}
                    style={{
                      flex: `${r.costUsd} 0 0`,
                      background: "#4a71d2",
                    }}
                  />
                  {r.cpuUsd && (
                    <span
                      className="ar-nd-vseg"
                      style={{
                        flex: `0 0 ${Math.max(3, Math.round(r.cpuUsd * PX_PER_USD))}px`,
                        background: "#d4a000",
                      }}
                    />
                  )}
                </div>
                <div className="ar-nd-cell ar-nd-cost" role="cell">
                  <strong>{r.cost}</strong>
                  <span>{r.time}</span>
                </div>
                <BranchCell br={r.a} win={r.winner === "a"} />
                <BranchCell br={r.b} win={r.winner === "b"} />
                <div className="ar-nd-cell r ar-nd-strength" role="cell">
                  {r.best === "—" ? "—" : `+${r.best}`}
                </div>
              </div>
            );
          })}

          <div
            className="ar-nd-row ar-nd-final"
            role="row"
            style={{ minHeight: h(2.73) }}
          >
            <div className="ar-nd-bar-col" aria-hidden="true">
              <span
                className="ar-nd-vseg"
                style={{ flex: "1 0 0", background: "#19337d" }}
              />
            </div>
            <div className="ar-nd-cell ar-nd-cost" role="cell">
              <strong>$2.73</strong>
              <span>56m</span>
            </div>
            <div className="ar-nd-cell ar-nd-span2 ar-nd-win" role="cell">
              <span className="ar-nd-bname">
                1-hour validation of winning recipe
              </span>
              <span className="ar-nd-val-summary">
                Dual-view critic and turn-based credit, spatial pyramid,
                scoreboard inputs, historical opponent pool, and fast 1-epoch
                updates on smaller batches.
              </span>
              <span className="ar-nd-sub">
                Final submission · beat both starter benchmarks in practice
              </span>
            </div>
            <div
              className="ar-nd-cell r ar-nd-strength ar-nd-final-val"
              role="cell"
            >
              +47.9%
            </div>
          </div>

          <div className="ar-nd-row ar-nd-tail" role="row">
            <div className="ar-nd-bar-col" aria-hidden="true">
              <span
                className="ar-nd-vseg is-last"
                style={{
                  height: Math.max(5, h(UNUSED)),
                  background: "#dcdbd5",
                }}
              />
            </div>
            <div className="ar-nd-tail-label" role="cell">
              $0.12 unused (of $8.70)
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
