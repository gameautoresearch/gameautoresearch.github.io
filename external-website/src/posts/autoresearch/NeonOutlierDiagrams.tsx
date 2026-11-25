import { useAgentView } from "../../view";

const BLUE = "#2f55c4";
const BLUE_TINT = "#e8eefc";
const BLUE_MID = "#7f9ae0";
const OPP = "#e05a2b";
const INK = "#111111";
const MUTED = "#767670";
const LINE = "#d4d8e4";
const FOG = "#e8e8e4";

function PyramidDiagram() {
  const fine = 5; // 5x5 visual proxy for 11x11
  const coarse = 3; // 3x3 visual proxy for 6x6
  const cellF = 11;
  const cellC = 14;
  return (
    <svg
      viewBox="0 0 280 122"
      role="img"
      aria-label="11 by 11 board grid pools down to a 6 by 6 coarse grid for board-wide convolutions, then upsamples and adds back to the 11 by 11 grid"
    >
      <defs>
        <marker
          id="od-ah-pyr"
          viewBox="0 0 8 8"
          refX="7"
          refY="4"
          markerWidth="5.5"
          markerHeight="5.5"
          orient="auto"
        >
          <path d="M0 1L8 4L0 7Z" fill={BLUE} />
        </marker>
      </defs>

      {/* Left: 11x11 local */}
      <g transform="translate(14 22)">
        {Array.from({ length: fine * fine }, (_, idx) => {
          const r = Math.floor(idx / fine);
          const c = idx % fine;
          const local = Math.abs(r - 2) <= 1 && Math.abs(c - 2) <= 1;
          const center = r === 2 && c === 2;
          return (
            <rect
              key={idx}
              x={c * cellF}
              y={r * cellF}
              width={cellF}
              height={cellF}
              fill={center ? BLUE : local ? "#eceef5" : "#fff"}
              stroke={LINE}
              strokeWidth={0.8}
            />
          );
        })}
        <rect
          x={cellF}
          y={cellF}
          width={cellF * 3}
          height={cellF * 3}
          fill="none"
          stroke={MUTED}
          strokeWidth={1.1}
          strokeDasharray="2.5 2"
        />
        <text x={27.5} y={71} className="od-lbl mid">
          11×11
        </text>
        <text x={27.5} y={83} className="od-sub mid">
          local 3×3
        </text>
      </g>

      {/* Arrow down-right */}
      <path
        d="M74 49 L111 49"
        stroke={BLUE}
        strokeWidth={1.4}
        fill="none"
        markerEnd="url(#od-ah-pyr)"
      />
      <text x={93} y={42} className="od-sub mid" style={{ fill: BLUE }}>
        pool
      </text>

      {/* Middle: 6x6 coarse */}
      <g transform="translate(119 28)">
        {Array.from({ length: coarse * coarse }, (_, idx) => {
          const r = Math.floor(idx / coarse);
          const c = idx % coarse;
          const center = r === 1 && c === 1;
          return (
            <rect
              key={idx}
              x={c * cellC}
              y={r * cellC}
              width={cellC}
              height={cellC}
              fill={center ? BLUE : BLUE_TINT}
              stroke={BLUE_MID}
              strokeWidth={0.9}
            />
          );
        })}
        <rect
          x={0}
          y={0}
          width={cellC * coarse}
          height={cellC * coarse}
          fill="none"
          stroke={BLUE}
          strokeWidth={1.4}
        />
        <text x={21} y={65} className="od-lbl mid" style={{ fill: BLUE }}>
          6×6
        </text>
        <text x={21} y={77} className="od-sub mid">
          coarse convs
        </text>
      </g>

      {/* Arrow right */}
      <path
        d="M167 49 L204 49"
        stroke={BLUE}
        strokeWidth={1.4}
        fill="none"
        markerEnd="url(#od-ah-pyr)"
      />
      <text x={185} y={42} className="od-sub mid" style={{ fill: BLUE }}>
        upsample
      </text>

      {/* Right: 11x11 + global */}
      <g transform="translate(211 22)">
        {Array.from({ length: fine * fine }, (_, idx) => {
          const r = Math.floor(idx / fine);
          const c = idx % fine;
          const center = r === 2 && c === 2;
          return (
            <rect
              key={idx}
              x={c * cellF}
              y={r * cellF}
              width={cellF}
              height={cellF}
              fill={center ? BLUE : BLUE_TINT}
              stroke={BLUE_MID}
              strokeWidth={0.75}
            />
          );
        })}
        <rect
          x={0}
          y={0}
          width={cellF * fine}
          height={cellF * fine}
          fill="none"
          stroke={BLUE}
          strokeWidth={1.3}
        />
        <text x={27.5} y={71} className="od-lbl mid">
          11×11 + global
        </text>
        <text x={27.5} y={83} className="od-sub mid">
          full-map reach
        </text>
      </g>
    </svg>
  );
}

function ScoreboardDiagram() {
  const rows: { label: string; ours: number; theirs: number; delta: string }[] =
    [
      { label: "Cities", ours: 5, theirs: 3, delta: "+2" },
      { label: "Units", ours: 8, theirs: 5, delta: "+3" },
      { label: "Tech", ours: 6, theirs: 4, delta: "+2" },
      { label: "Score", ours: 7, theirs: 5, delta: "+620" },
    ];
  const tokens: [number, number, string][] = [
    [1, 1, BLUE],
    [1, 3, BLUE],
    [2, 2, BLUE],
    [3, 1, BLUE],
    [2, 4, OPP],
    [3, 4, OPP],
    [4, 3, OPP],
  ];
  return (
    <svg
      viewBox="0 0 280 122"
      role="img"
      aria-label="Visible board tiles are summed into a 36-scalar scoreboard of cities, units, tech, score, and relative leads"
    >
      <defs>
        <marker
          id="od-ah-sb"
          viewBox="0 0 8 8"
          refX="7"
          refY="4"
          markerWidth="5.5"
          markerHeight="5.5"
          orient="auto"
        >
          <path d="M0 1L8 4L0 7Z" fill={BLUE} />
        </marker>
      </defs>

      {/* Left: mini board with units/cities */}
      <g transform="translate(14 18)">
        {Array.from({ length: 25 }, (_, i) => {
          const r = Math.floor(i / 5);
          const c = i % 5;
          const fog = r + c >= 7;
          return (
            <rect
              key={i}
              x={c * 13}
              y={r * 13}
              width={13}
              height={13}
              fill={fog ? FOG : "#fff"}
              stroke={LINE}
              strokeWidth={0.8}
            />
          );
        })}
        {tokens.map(([r, c, color], i) => (
          <circle
            key={i}
            cx={c * 13 + 6.5}
            cy={r * 13 + 6.5}
            r={3.2}
            fill={color}
          />
        ))}
        <text x={32.5} y={81} className="od-lbl mid">
          Visible board
        </text>
        <text x={32.5} y={93} className="od-sub mid">
          121 tiles
        </text>
      </g>

      {/* Arrow */}
      <path
        d="M84 50 L114 50"
        stroke={BLUE}
        strokeWidth={1.4}
        fill="none"
        markerEnd="url(#od-ah-sb)"
      />
      <text x={99} y={43} className="od-sub mid" style={{ fill: BLUE }}>
        tally
      </text>

      {/* Right: scoreboard panel */}
      <g transform="translate(121 12)">
        <rect
          x={0}
          y={0}
          width={145}
          height={86}
          rx={4}
          fill="#fff"
          stroke={BLUE_MID}
          strokeWidth={1.1}
        />
        <rect x={0} y={0} width={145} height={18} rx={4} fill={BLUE_TINT} />
        <rect x={0} y={12} width={145} height={6} fill={BLUE_TINT} />
        <text
          x={8}
          y={12.5}
          className="od-tiny"
          style={{ fill: BLUE, fontWeight: 600 }}
        >
          36 SUMMARY SCALARS
        </text>
        {rows.map((row, idx) => {
          const y = 31 + idx * 15;
          return (
            <g key={row.label}>
              <text x={8} y={y} className="od-sub" style={{ fill: INK }}>
                {row.label}
              </text>
              <rect
                x={46}
                y={y - 7}
                width={row.ours * 5.2}
                height={4.2}
                rx={1.5}
                fill={BLUE}
              />
              <rect
                x={46}
                y={y - 1.5}
                width={row.theirs * 5.2}
                height={4.2}
                rx={1.5}
                fill={OPP}
                opacity={0.75}
              />
              <text
                x={137}
                y={y}
                className="od-sub"
                textAnchor="end"
                style={{ fill: BLUE, fontWeight: 600 }}
              >
                {row.delta}
              </text>
            </g>
          );
        })}
        <text x={72.5} y={100} className="od-sub mid">
          fed straight into global state
        </text>
      </g>
    </svg>
  );
}

function CriticDistillDiagram() {
  return (
    <svg
      viewBox="0 0 280 122"
      role="img"
      aria-label="Policy sees a fogged board to pick moves, while a training-only critic sees both sides without fog and distills its value signal into an auxiliary head on the policy"
    >
      <defs>
        <marker
          id="od-ah-ink"
          viewBox="0 0 8 8"
          refX="7"
          refY="4"
          markerWidth="5.5"
          markerHeight="5.5"
          orient="auto"
        >
          <path d="M0 1L8 4L0 7Z" fill={INK} />
        </marker>
        <marker
          id="od-ah-dist"
          viewBox="0 0 8 8"
          refX="7"
          refY="4"
          markerWidth="5.5"
          markerHeight="5.5"
          orient="auto"
        >
          <path d="M0 1L8 4L0 7Z" fill={BLUE} />
        </marker>
        <pattern
          id="od-fog-hatch"
          width="5"
          height="5"
          patternUnits="userSpaceOnUse"
          patternTransform="rotate(45)"
        >
          <rect width="5" height="5" fill={FOG} />
          <line
            x1="0"
            y1="0"
            x2="0"
            y2="5"
            stroke="#c8c8c2"
            strokeWidth="1.2"
          />
        </pattern>
      </defs>

      {/* Top row: Policy (fogged) */}
      <g transform="translate(14 14)">
        <rect
          x={0}
          y={0}
          width={34}
          height={34}
          rx={3}
          fill="#fff"
          stroke={LINE}
        />
        <rect
          x={17}
          y={0}
          width={17}
          height={34}
          rx={2}
          fill="url(#od-fog-hatch)"
        />
        <circle cx={9} cy={13} r={3} fill={BLUE} />
        <circle cx={11} cy={24} r={3} fill={BLUE} />
        <text x={25.5} y={20.5} className="od-lbl mid" style={{ fill: MUTED }}>
          ?
        </text>
        <text x={17} y={45} className="od-tiny mid">
          fogged
        </text>
      </g>

      <path
        d="M52 31 L78 31"
        stroke={INK}
        strokeWidth={1.3}
        fill="none"
        markerEnd="url(#od-ah-ink)"
      />

      {/* Policy box + aux head */}
      <rect
        x={82}
        y={13}
        width={94}
        height={36}
        rx={4}
        fill={BLUE_TINT}
        stroke={BLUE}
        strokeWidth={1.2}
      />
      <text x={129} y={28} className="od-lbl mid" style={{ fill: BLUE }}>
        Policy (4×64)
      </text>
      <rect
        x={96}
        y={34}
        width={66}
        height={12}
        rx={2.5}
        fill="#fff"
        stroke={BLUE_MID}
        strokeWidth={0.9}
      />
      <text x={129} y={42.5} className="od-tiny mid" style={{ fill: BLUE }}>
        aux value head
      </text>

      <path
        d="M180 31 L212 31"
        stroke={INK}
        strokeWidth={1.3}
        fill="none"
        markerEnd="url(#od-ah-ink)"
      />
      <rect
        x={216}
        y={19}
        width={50}
        height={24}
        rx={4}
        fill="#fff"
        stroke={INK}
        strokeWidth={1.1}
      />
      <text x={241} y={34} className="od-lbl mid">
        Move
      </text>

      {/* Bottom row: Omniscient Critic */}
      <g transform="translate(14 68)">
        <rect
          x={0}
          y={0}
          width={34}
          height={34}
          rx={3}
          fill="#fff"
          stroke={BLUE_MID}
        />
        <circle cx={9} cy={13} r={3} fill={BLUE} />
        <circle cx={11} cy={24} r={3} fill={BLUE} />
        <circle cx={24} cy={11} r={3} fill={OPP} />
        <circle cx={25} cy={23} r={3} fill={OPP} />
        <text x={17} y={45} className="od-tiny mid" style={{ fill: BLUE }}>
          both views
        </text>
      </g>

      <path
        d="M52 85 L78 85"
        stroke={BLUE}
        strokeWidth={1.3}
        fill="none"
        markerEnd="url(#od-ah-dist)"
      />

      <rect
        x={82}
        y={69}
        width={94}
        height={32}
        rx={4}
        fill="#fff"
        stroke={BLUE}
        strokeWidth={1.2}
        strokeDasharray="3.5 2.5"
      />
      <text x={129} y={83} className="od-lbl mid">
        Critic (2×48)
      </text>
      <text x={129} y={95} className="od-tiny mid">
        training only
      </text>

      <path
        d="M180 85 L212 85"
        stroke={BLUE}
        strokeWidth={1.3}
        fill="none"
        markerEnd="url(#od-ah-dist)"
      />
      <rect
        x={216}
        y={73}
        width={50}
        height={24}
        rx={4}
        fill={BLUE_TINT}
        stroke={BLUE}
        strokeWidth={1.1}
      />
      <text
        x={241}
        y={88}
        className="od-sub mid"
        style={{ fill: BLUE, fontWeight: 600 }}
      >
        Value
      </text>

      {/* Distillation arrow from Value -> aux value head */}
      <path
        d="M241 72 C241 56, 172 60, 152 49"
        stroke={BLUE}
        strokeWidth={1.4}
        strokeDasharray="3 2"
        fill="none"
        markerEnd="url(#od-ah-dist)"
      />
      <text
        x={202}
        y={61}
        className="od-tiny mid"
        style={{
          fill: BLUE,
          fontWeight: 600,
          paintOrder: "stroke",
          stroke: "#fff",
          strokeWidth: 3.5,
          strokeLinejoin: "round",
        }}
      >
        distill
      </text>
    </svg>
  );
}

function PfspDiagram() {
  const pool = [
    { id: "v2", wr: "84%", status: "evict" as const },
    { id: "v7", wr: "61%", status: "mid" as const },
    { id: "v14", wr: "43%", status: "hard" as const },
    { id: "v19", wr: "49%", status: "hard" as const },
  ];
  return (
    <svg
      viewBox="0 0 280 122"
      role="img"
      aria-label="75 percent of games are self-play against the current bot; 25 percent sample from a 20-checkpoint pool weighted toward tough past copies while evicting easy ones"
    >
      <defs>
        <marker
          id="od-ah-pf"
          viewBox="0 0 8 8"
          refX="7"
          refY="4"
          markerWidth="5.5"
          markerHeight="5.5"
          orient="auto"
        >
          <path d="M0 1L8 4L0 7Z" fill={BLUE} />
        </marker>
        <marker
          id="od-ah-pfk"
          viewBox="0 0 8 8"
          refX="7"
          refY="4"
          markerWidth="5.5"
          markerHeight="5.5"
          orient="auto"
        >
          <path d="M0 1L8 4L0 7Z" fill={INK} />
        </marker>
      </defs>

      {/* Learner box */}
      <rect x={12} y={39} width={62} height={36} rx={4} fill={BLUE} />
      <text x={43} y={55} className="od-lbl mid" style={{ fill: "#fff" }}>
        Current
      </text>
      <text x={43} y={67} className="od-tiny mid" style={{ fill: "#dbe5ff" }}>
        learner
      </text>

      {/* Top branch: 75% vs Self */}
      <path
        d="M74 49 C92 49, 92 26, 114 26"
        stroke={INK}
        strokeWidth={1.6}
        fill="none"
        markerEnd="url(#od-ah-pfk)"
      />
      <text
        x={93}
        y={24}
        className="od-tiny mid"
        style={{ fill: INK, fontWeight: 600 }}
      >
        75%
      </text>
      <rect
        x={118}
        y={13}
        width={148}
        height={25}
        rx={4}
        fill="#fff"
        stroke={LINE}
        strokeWidth={1.1}
      />
      <text x={192} y={29} className="od-lbl mid">
        vs. current self
      </text>

      {/* Bottom branch: 25% vs PFSP pool */}
      <path
        d="M74 65 C92 65, 92 78, 114 78"
        stroke={BLUE}
        strokeWidth={1.6}
        fill="none"
        markerEnd="url(#od-ah-pf)"
      />
      <text
        x={93}
        y={88}
        className="od-tiny mid"
        style={{ fill: BLUE, fontWeight: 600 }}
      >
        25%
      </text>

      {/* Pool container */}
      <rect
        x={118}
        y={48}
        width={148}
        height={58}
        rx={4}
        fill="#fff"
        stroke={BLUE_MID}
        strokeWidth={1.1}
      />
      <text
        x={125}
        y={59}
        className="od-tiny"
        style={{ fill: BLUE, fontWeight: 600 }}
      >
        20-CHECKPOINT POOL
      </text>

      {pool.map((p, i) => {
        const x = 125 + i * 34;
        const hard = p.status === "hard";
        const evict = p.status === "evict";
        return (
          <g key={p.id} transform={`translate(${x} 67)`}>
            <rect
              x={0}
              y={0}
              width={29}
              height={32}
              rx={3}
              fill={hard ? BLUE_TINT : evict ? "#f7f7f5" : "#fff"}
              stroke={hard ? BLUE : evict ? "#d0d0ca" : LINE}
              strokeWidth={hard ? 1.5 : 1}
              strokeDasharray={evict ? "2.5 2" : undefined}
            />
            <text
              x={14.5}
              y={13}
              className="od-tiny mid"
              style={{
                fill: evict ? "#a0a09a" : hard ? BLUE : INK,
                fontWeight: hard ? 700 : 600,
              }}
            >
              {p.id}
            </text>
            <text
              x={14.5}
              y={25}
              className="od-tiny mid"
              style={{ fill: evict ? OPP : hard ? BLUE : MUTED }}
            >
              {p.wr}
            </text>
            {evict && (
              <g transform="translate(27 2)">
                <circle cx={0} cy={0} r={4.6} fill={OPP} />
                <path
                  d="M-1.9 -1.9 L1.9 1.9 M1.9 -1.9 L-1.9 1.9"
                  stroke="#fff"
                  strokeWidth={1.25}
                  strokeLinecap="round"
                />
              </g>
            )}
          </g>
        );
      })}
      <text x={192} y={115} className="od-tiny mid">
        <tspan fill={OPP}>evict easy (&gt;78%)</tspan>
        <tspan fill={MUTED}> · </tspan>
        <tspan fill={BLUE}>weight tough copies</tspan>
      </text>
    </svg>
  );
}

const ITEMS = [
  {
    key: "pyramid",
    title: "Cheap full-board context",
    text: "A coarse spatial pyramid gives every tile board-wide awareness at little cost.",
    Diagram: PyramidDiagram,
  },
  {
    key: "scoreboard",
    title: "Explicit scoreboard summaries",
    text: "Tallies of the visible board (cities, units, HP, territory) and each side's relative lead, fed in as direct inputs.",
    Diagram: ScoreboardDiagram,
  },
  {
    key: "critic",
    title: "Omniscient critic distilled into the policy",
    text: "A training critic value head sees both sides through the fog of war. We distill its judgment into the policy.",
    Diagram: CriticDistillDiagram,
  },
  {
    key: "pfsp",
    title: "Pruned historical self-play",
    text: "A pool of past checkpoints, pruned to the ones the learner still struggles against.",
    Diagram: PfspDiagram,
  },
] as const;

export default function NeonOutlierDiagrams() {
  if (useAgentView()) {
    return (
      <ul className="ar-outlier-list">
        {ITEMS.map((item) => (
          <li key={item.key}>
            <strong>{item.title}:</strong> {item.text}
          </li>
        ))}
      </ul>
    );
  }
  return (
    <div className="ar-od-grid">
      {ITEMS.map(({ key, title, text, Diagram }) => (
        <div key={key} className="ar-od-card">
          <strong className="ar-od-title">{title}</strong>
          <div className="ar-od-svg">
            <Diagram />
          </div>
          <p className="ar-od-text">{text}</p>
        </div>
      ))}
    </div>
  );
}
