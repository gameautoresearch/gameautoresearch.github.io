import { useAgentView } from "../../view";

const C = "#00a0d6";
const C_DEEP = "#0081ad";
const C_TINT = "#e3f5fb";
const C_MID = "#7fcfea";

const M = "#e0218a";
const M_DEEP = "#b8156e";
const M_TINT = "#fde6f1";
const M_MID = "#f09bc8";

const INK = "#111111";
const MUTED = "#6b6b6b";
const RULE = "#d4d4ce";

function TrainedModelDiagram() {
  const probes = [
    { x: 13, d: "M16 49 L20 44 L23.5 46 L28 36" },
    { x: 34, d: "M37 49 L41 41 L44.5 43 L49 39" },
    { x: 55, d: "M58 49 L62 40 L65.5 37 L70 32" },
  ];
  const l1 = [32, 42, 52];
  const l2 = [28.5, 37.5, 46.5, 55.5];
  const l3 = [37, 47];

  return (
    <svg
      viewBox="0 0 300 108"
      role="img"
      aria-label="One shared agent compute budget is split between short test runs and a final training run; the trained model weights are submitted directly for evaluation."
    >
      <defs>
        <marker
          id="sf-ah-c"
          viewBox="0 0 8 8"
          refX="7"
          refY="4"
          markerWidth="5.5"
          markerHeight="5.5"
          orient="auto"
        >
          <path d="M0 1L8 4L0 7Z" fill={C_DEEP} />
        </marker>
      </defs>

      {/* Column headers */}
      <text x={8} y={11} className="sf-kicker" style={{ fill: C_DEEP }}>
        ONE SHARED BUDGET
      </text>
      <text x={203} y={11} className="sf-kicker mid">
        SUBMIT
      </text>
      <text x={260} y={11} className="sf-kicker mid">
        EVAL
      </text>

      {/* Submission boundary */}
      <line
        x1={203}
        y1={16}
        x2={203}
        y2={84}
        stroke={RULE}
        strokeWidth={1}
        strokeDasharray="3 2.5"
      />

      {/* Shared agent compute budget box */}
      <rect
        x={6}
        y={18}
        width={176}
        height={64}
        rx={5}
        fill="#fff"
        stroke={C_MID}
        strokeWidth={1.1}
      />

      {/* Short research test runs */}
      {probes.map((p, i) => (
        <g key={i}>
          <rect
            x={p.x}
            y={26}
            width={18}
            height={28}
            rx={2.5}
            fill={C_TINT}
            stroke={C}
            strokeWidth={0.95}
          />
          <path
            d={p.d}
            fill="none"
            stroke={C_DEEP}
            strokeWidth={1.15}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </g>
      ))}

      {/* Trade-off split between tests and final run */}
      <line
        x1={83}
        y1={23}
        x2={83}
        y2={57}
        stroke={C_DEEP}
        strokeWidth={1}
        strokeDasharray="2.5 2"
        opacity={0.55}
      />
      <path
        d="M80 37.5 L77 40 L80 42.5 M86 37.5 L89 40 L86 42.5"
        fill="none"
        stroke={C_DEEP}
        strokeWidth={1.15}
        strokeLinecap="round"
        strokeLinejoin="round"
      />

      {/* Final training run carved out of the same budget */}
      <rect x={93} y={26} width={82} height={28} rx={2.5} fill={C} />
      <path
        d="M98 50 C116 48, 136 37, 170 30"
        fill="none"
        stroke="#fff"
        strokeWidth={1.3}
        strokeLinecap="round"
        opacity={0.22}
      />
      <text x={134} y={43.5} className="sf-lbl mid" style={{ fill: "#fff" }}>
        Final train
      </text>

      <text x={43} y={72} className="sf-sub mid">
        Tests
      </text>
      <text
        x={134}
        y={72}
        className="sf-sub mid"
        style={{ fill: C_DEEP, fontWeight: 600 }}
      >
        Rest of budget
      </text>

      {/* Submission arrow carrying model weights */}
      <path
        d="M183 50 L221 50"
        stroke={C_DEEP}
        strokeWidth={1.4}
        fill="none"
        markerEnd="url(#sf-ah-c)"
      />
      <text
        x={203}
        y={43}
        className="sf-sub mid sf-halo"
        style={{ fill: C_DEEP, fontWeight: 600 }}
      >
        weights
      </text>

      {/* Evaluation card with mini neural network */}
      <rect
        x={226}
        y={18}
        width={68}
        height={64}
        rx={5}
        fill={C_TINT}
        stroke={C}
        strokeWidth={1.1}
      />
      <g>
        {l1.map((y1, i) =>
          l2.map((y2, j) => (
            <line
              key={`a-${i}-${j}`}
              x1={244}
              y1={y1}
              x2={260}
              y2={y2}
              stroke={C_DEEP}
              strokeWidth={0.65}
              opacity={0.45}
            />
          )),
        )}
        {l2.map((y2, i) =>
          l3.map((y3, j) => (
            <line
              key={`b-${i}-${j}`}
              x1={260}
              y1={y2}
              x2={276}
              y2={y3}
              stroke={C_DEEP}
              strokeWidth={0.65}
              opacity={0.45}
            />
          )),
        )}
        {l1.map((y, i) => (
          <circle
            key={`n1-${i}`}
            cx={244}
            cy={y}
            r={2.3}
            fill="#fff"
            stroke={C_DEEP}
            strokeWidth={1.1}
          />
        ))}
        {l2.map((y, i) => (
          <circle
            key={`n2-${i}`}
            cx={260}
            cy={y}
            r={2.3}
            fill="#fff"
            stroke={C_DEEP}
            strokeWidth={1.1}
          />
        ))}
        {l3.map((y, i) => (
          <circle
            key={`n3-${i}`}
            cx={276}
            cy={y}
            r={2.3}
            fill="#fff"
            stroke={C_DEEP}
            strokeWidth={1.1}
          />
        ))}
      </g>
      <text
        x={260}
        y={72}
        className="sf-sub mid"
        style={{ fill: C_DEEP, fontWeight: 600 }}
      >
        Weights
      </text>

      {/* Footer captions */}
      <text x={94} y={98} className="sf-cap mid">
        Tests &amp; final run share one budget
      </text>
      <text x={260} y={98} className="sf-cap mid">
        Graded as-is
      </text>
    </svg>
  );
}

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
      viewBox="0 0 300 108"
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

      {/* Footer captions */}
      <text x={69} y={98} className="sf-cap mid">
        No final run to save for
      </text>
      <text x={241} y={98} className="sf-cap mid">
        Retrained from scratch
      </text>
    </svg>
  );
}

const FORMATS = [
  {
    key: "model",
    title: "Submit a trained model",
    body: "The agent trains the final network itself and submits the weights. Tests and the final run share one budget.",
    Diagram: TrainedModelDiagram,
  },
  {
    key: "recipe",
    title: "Submit a training recipe",
    body: "The agent submits training code, and we retrain it from scratch on a fixed budget. The agent’s entire budget goes to research.",
    Diagram: TrainingRecipeDiagram,
  },
] as const;

export default function SubmissionFormats() {
  if (useAgentView()) {
    return (
      <ul>
        {FORMATS.map((f) => (
          <li key={f.key}>
            <strong>{f.title}.</strong> {f.body}
          </li>
        ))}
      </ul>
    );
  }

  return (
    <div className="ar-configs">
      {FORMATS.map(({ key, title, body, Diagram }) => (
        <div key={key} className={`ar-sf-card ar-sf-${key}`}>
          <h3>{title}</h3>
          <div className="ar-sf-svg">
            <Diagram />
          </div>
          <p>{body}</p>
        </div>
      ))}
    </div>
  );
}
