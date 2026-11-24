import { useAgentView } from "../../view";
import { Table } from "../shared/text";

// Gemini 4 Neon's $8.70 Polytopia recipe session (`polytopia-neon-b3`, $8.58 spent).
// Sourced from its research ledger (`ledger.sqlite3`) and `NOTES.md`.

type Arm = {
  label: string;
  score: number | null; // practice win rate %, null if crashed
  crash?: string;
  winner?: boolean;
};

type Stage = {
  key: string;
  badge: string;
  title: string;
  cost: string;
  color: string;
  arms: [Arm, Arm];
  note: string;
};

const BUDGET = 8.7;
const SPENT = 8.58;
const VALIDATION_COST = 2.73;

const STAGES: Stage[] = [
  {
    key: "r1",
    badge: "A/B 1",
    title: "Baseline vs. dual-view critic and turn-based credit",
    cost: "$0.90",
    color: "#a8beee",
    arms: [
      { label: "Starter recipe", score: 0.0 },
      { label: "Dual-view critic + turn-based credit", score: 3.1, winner: true },
    ],
    note: "Beat the starter in a short run and revealed that slow training speed was the main bottleneck.",
  },
  {
    key: "r2",
    badge: "A/B 2",
    title: "Faster pyramid network: self-play vs. opponent pool",
    cost: "$1.23 + $0.04 CPU",
    color: "#8aa8e8",
    arms: [
      { label: "Pyramid network, self-play only", score: null, crash: "Eval crash" },
      { label: "Pyramid network + opponent pool", score: null, crash: "Eval crash" },
    ],
    note: "Instead of guessing on GPUs, Neon spent $0.04 on six tiny CPU-only workers to isolate an eval crash in the new network and a bug in the opponent pool.",
  },
  {
    key: "r3",
    badge: "A/B 3",
    title: "Self-play vs. historical opponent pool",
    cost: "$1.22",
    color: "#6a8ee0",
    arms: [
      { label: "Self-play only", score: 11.3 },
      { label: "With historical opponent pool", score: 15.2, winner: true },
    ],
    note: "Clean single-variable check after the fixes: playing past versions of itself clearly helped.",
  },
  {
    key: "r4",
    badge: "A/B 4",
    title: "Faster updates vs. a richer critic",
    cost: "$1.23",
    color: "#4a71d2",
    arms: [
      { label: "1-epoch updates, standard critic", score: 18.4, winner: true },
      { label: "2-epoch updates, distributional critic", score: 15.4 },
    ],
    note: "Added scoreboard inputs; faster throughput beat the heavier critic and logs showed room for more frequent gradient steps.",
  },
  {
    key: "r5",
    badge: "A/B 5",
    title: "More frequent updates vs. heavier shaping",
    cost: "$1.23",
    color: "#2f55c4",
    arms: [
      { label: "Smaller batches (more updates)", score: 22.7, winner: true },
      { label: "Larger batches + heavier shaping", score: 21.7 },
    ],
    note: "Doubled gradient step frequency without losing speed; smaller batches won overall and balanced both seats.",
  },
];

const BAR_SEGMENTS = [
  { key: "r1", label: "A/B 1", cost: 0.9, color: "#a8beee" },
  { key: "r2", label: "A/B 2", cost: 1.23, color: "#8aa8e8" },
  { key: "dbg", label: "CPU debug", cost: 0.04, color: "#d4a000" },
  { key: "r3", label: "A/B 3", cost: 1.22, color: "#6a8ee0" },
  { key: "r4", label: "A/B 4", cost: 1.23, color: "#4a71d2" },
  { key: "r5", label: "A/B 5", cost: 1.23, color: "#2f55c4" },
  { key: "val", label: "1 h validation", cost: VALIDATION_COST, color: "#19337d" },
  { key: "left", label: "Unused", cost: BUDGET - SPENT, color: "#e6e5e0" },
];

const MAX_SHORT_SCORE = 25;

export default function NeonDiscipline() {
  if (useAgentView()) {
    return (
      <Table
        head={["Stage", "Cost", "Arm A", "Arm B", "Takeaway"]}
        rows={[
          ...STAGES.map((s) => [
            `${s.badge}: ${s.title}`,
            s.cost,
            `${s.arms[0].label}: ${s.arms[0].score !== null ? `${s.arms[0].score.toFixed(1)}%` : s.arms[0].crash}`,
            `${s.arms[1].label}: ${s.arms[1].score !== null ? `${s.arms[1].score.toFixed(1)}%` : s.arms[1].crash}`,
            s.note,
          ]),
          [
            "Validation",
            "$2.73",
            "Winning recipe trained for 1 hour: 47.9%",
            "—",
            "Confirmed end-to-end before submission ($8.58 of $8.70 spent); reached 416 Elo at 8 h host retrain.",
          ],
        ]}
      />
    );
  }

  return (
    <div className="ar-nd">
      <div className="ar-nd-budget">
        <div className="ar-nd-budget-head">
          <strong>$8.70 research budget</strong>
          <span>5 short A/B tests ($5.85) · 1 h validation ($2.73)</span>
        </div>
        <div
          className="ar-nd-bar"
          role="img"
          aria-label="Breakdown of Gemini 4 Neon's $8.70 budget: five short A/B tests totaling $5.81, $0.04 of CPU debugging, $2.73 for a 1-hour validation run, and $0.12 unused"
        >
          {BAR_SEGMENTS.map((seg) => {
            const pct = (seg.cost / BUDGET) * 100;
            return (
              <div
                key={seg.key}
                className="ar-nd-seg"
                style={{
                  width: `${Math.max(pct, seg.key === "dbg" ? 0.7 : 0)}%`,
                  background: seg.color,
                }}
              >
                {pct >= 10 && (
                  <span className={seg.key === "r1" ? "dark" : ""}>
                    {seg.key === "val" ? "1 h validation · $2.73" : seg.label}
                  </span>
                )}
              </div>
            );
          })}
        </div>
        <div className="ar-nd-bar-foot">
          <span>
            <i style={{ background: "#4a71d2" }} />5 A/B tests ($5.81)
          </span>
          <span>
            <i style={{ background: "#d4a000" }} />CPU debug ($0.04)
          </span>
          <span>
            <i style={{ background: "#19337d" }} />1 h validation ($2.73)
          </span>
          <span>
            <i style={{ background: "#e6e5e0" }} />Unused ($0.12)
          </span>
        </div>
      </div>

      <div className="ar-nd-list">
        {STAGES.map((stage) => (
          <div key={stage.key} className="ar-nd-stage">
            <div className="ar-nd-stage-head">
              <div className="ar-nd-stage-title">
                <span
                  className="ar-nd-badge"
                  style={{
                    background: stage.color,
                    color: stage.key === "r1" ? "#102154" : "#fff",
                  }}
                >
                  {stage.badge}
                </span>
                <strong>{stage.title}</strong>
              </div>
              <span className="ar-nd-meta">{stage.cost}</span>
            </div>

            <div className="ar-nd-arms">
              {stage.arms.map((arm) => {
                const crashed = arm.score === null;
                return (
                  <div
                    key={arm.label}
                    className={`ar-nd-arm${arm.winner ? " is-win" : ""}${crashed ? " is-crash" : ""}`}
                  >
                    <div className="ar-nd-arm-top">
                      <span className="ar-nd-arm-label">
                        {arm.label}
                        {arm.winner && <em className="ar-nd-tag-win">kept</em>}
                      </span>
                      <span className="ar-nd-score">
                        {crashed ? arm.crash : `${arm.score!.toFixed(1)}%`}
                      </span>
                    </div>
                    {!crashed && (
                      <div className="ar-nd-meter" aria-hidden="true">
                        <i
                          style={{
                            width: `${Math.max((arm.score! / MAX_SHORT_SCORE) * 100, arm.score! > 0 ? 3 : 0)}%`,
                          }}
                        />
                      </div>
                    )}
                  </div>
                );
              })}
            </div>

            <p className="ar-nd-takeaway">{stage.note}</p>
          </div>
        ))}

        <div className="ar-nd-stage ar-nd-val">
          <div className="ar-nd-stage-head">
            <div className="ar-nd-stage-title">
              <span
                className="ar-nd-badge"
                style={{ background: "#19337d", color: "#fff" }}
              >
                Validation
              </span>
              <strong>1-hour run of the winning recipe</strong>
            </div>
            <span className="ar-nd-meta">$2.73</span>
          </div>
          <div className="ar-nd-val-body">
            <p className="ar-nd-val-desc">
              Scaled the winner of A/B 5 to the full 1-hour grading length and
              verified it beat both starter benchmarks before submitting.
            </p>
            <div className="ar-nd-val-stat">
              <strong>47.9%</strong>
              <span>practice win rate</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
