import { useAgentView } from "../../view";

const M = "#e0218a";
const M_DEEP = "#b8156e";
const M_TINT = "#fde6f1";
const M_MID = "#f09bc8";

const INK = "#111111";
const MUTED = "#6b6b6b";
const RULE = "#d4d4ce";

function TrainingRecipeDiagram() {
  const probes = [
    { x: 13, d: "M16 49 L20 44 L24 46 L30 37" },
    { x: 36, d: "M39 49 L43 42 L47 44 L53 39" },
    { x: 59, d: "M62 49 L66 40 L70 38 L76 34" },
    { x: 82, d: "M85 49 L89 43 L93 37 L99 32" },
    { x: 105, d: "M108 49 L112 39 L116 35 L122 30" },
  ];

  return (
    <svg
      viewBox="0 0 300 80"
      role="img"
      aria-label="The agent spends 100 percent of its compute budget on research tests and submits training code and settings; the benchmark then runs a fresh retrain from scratch on a separate fixed compute budget."
    >
      <defs>
        <marker
          id="sf-ah-m"
          viewBox="0 0 8 8"
          refX="7"
          refY="4"
          markerWidth="5.5"
          markerHeight="5.5"
          orient="auto"
        >
          <path d="M0 1L8 4L0 7Z" fill={M_DEEP} />
        </marker>
      </defs>

      {/* Column headers */}
      <text x={8} y={11} className="sf-kicker" style={{ fill: M_DEEP }}>
        RESEARCH BUDGET
      </text>
      <text x={160} y={11} className="sf-kicker mid">
        SUBMIT
      </text>
      <text x={190} y={11} className="sf-kicker">
        RETRAIN + EVAL
      </text>

      {/* Submission boundary */}
      <line
        x1={160}
        y1={16}
        x2={160}
        y2={84}
        stroke={RULE}
        strokeWidth={1}
        strokeDasharray="3 2.5"
      />

      {/* Agent research-only compute budget box */}
      <rect
        x={6}
        y={18}
        width={126}
        height={64}
        rx={5}
        fill="#fff"
        stroke={M_MID}
        strokeWidth={1.1}
      />

      {/* Five test runs filling the entire budget */}
      {probes.map((p, i) => (
        <g key={i}>
          <rect
            x={p.x}
            y={26}
            width={20}
            height={28}
            rx={2.5}
            fill={M_TINT}
            stroke={M}
            strokeWidth={0.95}
          />
          <path
            d={p.d}
            fill="none"
            stroke={M_DEEP}
            strokeWidth={1.15}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </g>
      ))}
      <text
        x={69}
        y={72}
        className="sf-sub mid"
        style={{ fill: M_DEEP, fontWeight: 600 }}
      >
        Tests only
      </text>

      {/* Submission arrow carrying code + config */}
      <path
        d="M134 50 L183 50"
        stroke={M_DEEP}
        strokeWidth={1.4}
        fill="none"
        markerEnd="url(#sf-ah-m)"
      />
      {/* Miniature recipe/code document icon */}
      <g>
        <path
          d="M151 24 Q151 22 153 22 H164 L169 27 V41 Q169 43 167 43 H153 Q151 43 151 41 Z"
          fill="#fff"
          stroke={M_DEEP}
          strokeWidth={1.1}
          strokeLinejoin="round"
        />
        <path
          d="M164 22 V27 H169"
          fill={M_TINT}
          stroke={M_DEEP}
          strokeWidth={0.9}
          strokeLinejoin="round"
        />
        <line
          x1={154.5}
          y1={31}
          x2={164.5}
          y2={31}
          stroke={M}
          strokeWidth={1.2}
          strokeLinecap="round"
        />
        <line
          x1={154.5}
          y1={35}
          x2={162}
          y2={35}
          stroke={MUTED}
          strokeWidth={1.1}
          strokeLinecap="round"
        />
        <line
          x1={154.5}
          y1={39}
          x2={165.5}
          y2={39}
          stroke={M}
          strokeWidth={1.2}
          strokeLinecap="round"
        />
      </g>
      <text
        x={160}
        y={63}
        className="sf-sub mid sf-halo"
        style={{ fill: M_DEEP, fontWeight: 600 }}
      >
        code
      </text>

      {/* Separate benchmark retrain container */}
      <rect
        x={188}
        y={18}
        width={106}
        height={64}
        rx={5}
        fill="#fafaf8"
        stroke="#b6b6b0"
        strokeWidth={1.1}
        strokeDasharray="3.5 2.5"
      />
      <rect x={196} y={26} width={90} height={28} rx={2.5} fill={M} />
      <path
        d="M201 50 C220 48, 240 37, 281 30"
        fill="none"
        stroke="#fff"
        strokeWidth={1.3}
        strokeLinecap="round"
        opacity={0.22}
      />
      <text x={241} y={43.5} className="sf-lbl mid" style={{ fill: "#fff" }}>
        Fresh train
      </text>
      <text
        x={241}
        y={72}
        className="sf-sub mid"
        style={{ fill: INK, fontWeight: 600 }}
      >
        Fixed budget
      </text>
    </svg>
  );
}

const TITLE = "Submit a training recipe";
const BODY =
  "The agent spends its whole budget on research and submits training code. We retrain it from scratch on a separate, fixed training budget and grade that network.";

export default function SubmissionFormats() {
  if (useAgentView()) {
    return (
      <p>
        <strong>{TITLE}.</strong> {BODY}
      </p>
    );
  }

  return (
    <div className="ar-sf-card ar-sf-recipe ar-sf-single">
      <h3>{TITLE}</h3>
      <div className="ar-sf-svg">
        <TrainingRecipeDiagram />
      </div>
    </div>
  );
}
