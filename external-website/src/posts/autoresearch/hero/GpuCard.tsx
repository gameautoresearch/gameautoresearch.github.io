import { useId } from "react";

const BLADE_ANGLES = Array.from({ length: 9 }, (_, i) => i * 40);
const GOLD_PINS = [...[42, 48, 54, 60], ...Array.from({ length: 18 }, (_, i) => 79 + i * 7)];

const clamp = (v: number, a = 0, b = 1) => Math.max(a, Math.min(b, v));

// Smooth linear interpolation between two #rrggbb colors.
function mix(a: string, b: string, k: number) {
  const t = clamp(k);
  const pa = parseInt(a.slice(1), 16), pb = parseInt(b.slice(1), 16);
  const ch = (shift: number) => Math.round(((pa >> shift) & 255) + (((pb >> shift) & 255) - ((pa >> shift) & 255)) * t);
  return `rgb(${ch(16)},${ch(8)},${ch(0)})`;
}

export default function GpuCard({ x, y, activity, spin, intake }: {
  x: number; y: number; activity: number; spin: number; intake: number;
}) {
  const id = useId();
  const paint = (name: string) => `url(#${id}-${name})`;
  const a = clamp(activity);
  const p = clamp(intake);
  const pulse = p > 0 && p < 1 ? Math.sin(p * Math.PI) ** 2 : 0;
  const rippleEase = 1 - (1 - p) ** 2;

  // Balanced editorial hardware palette: crisp depth without overpowering the bills and chart.
  const glow = mix("#8b948e", "#2aa95f", a);
  const shroudStroke = mix("#727974", "#3b9f66", a);
  const topTrim = mix("#7d8580", "#698173", a);
  const rimFill = mix("#dadfda", "#cfe3d7", a);
  const rimStroke = mix("#9aa39d", "#74ae8d", a);
  const hubStroke = mix("#747d77", "#56916f", a);
  const finStroke = mix("#8c958f", "#689f7e", a);
  const pcbFill = mix("#7a9183", "#649077", a);
  const pcbStroke = mix("#69766e", "#4e7c62", a);
  const badgeStroke = mix("#b5beb8", "#89bc9e", a);

  return (
    <g transform={`translate(${x} ${y + pulse * 0.45}) scale(.415)`}>
      <defs>
        <linearGradient id={`${id}-shroud`} x2="0" y2="100%">
          <stop stopColor={mix("#f7f9f7", "#f2faf5", a)} />
          <stop offset=".55" stopColor={mix("#eef0ed", "#e6f4eb", a)} />
          <stop offset="1" stopColor={mix("#e1e5e1", "#d7ebde", a)} />
        </linearGradient>
        <linearGradient id={`${id}-fan`} x2="0" y2="100%">
          <stop stopColor={mix("#666e69", "#5c7064", a)} />
          <stop offset="1" stopColor={mix("#7e8781", "#73887c", a)} />
        </linearGradient>
        <linearGradient id={`${id}-blade`} x2="0" y2="100%">
          <stop stopColor={mix("#b5beb8", "#a8bdb0", a)} />
          <stop offset="1" stopColor={mix("#88928c", "#7c9185", a)} />
        </linearGradient>
        <linearGradient id={`${id}-gold`} x2="0" y2="100%">
          <stop stopColor="#e2ce98" />
          <stop offset="1" stopColor="#c7ac6b" />
        </linearGradient>
        <filter id={`${id}-shadow`} x="-14%" y="-24%" width="128%" height="152%">
          <feDropShadow dx="0" dy="5" stdDeviation="5.5" floodColor="#18251d" floodOpacity=".12" />
        </filter>
      </defs>

      {/* Top auxiliary power connector */}
      <rect x="214" y="3" width="32" height="11" rx="2.5" fill="#828984" stroke="#686f6a" strokeWidth="1.4" />
      {[220, 226, 232, 238].map((pin) => <circle key={pin} cx={pin} cy="7.5" r="1.4" fill="#d8bf7f" />)}

      {/* Copper heatpipe loops */}
      <path d="M76 12C76 4.5 98 4.5 98 12M108 12C108 4.5 130 4.5 130 12" fill="none" stroke="#c79e7c" strokeWidth="3" strokeLinecap="round" />

      {/* PCIe I/O bracket */}
      <path d="M0 1.5Q0 0 1.5 0H18.5Q20 0 20 1.5V140Q20 142 18 142H12Q10 142 10 140V10H1.5Q0 10 0 8.5Z"
        fill="#e3e6e2" stroke="#717873" strokeWidth="2" strokeLinejoin="round" />

      {/* PCIe edge connector */}
      <path d="M38 128H66V139Q66 142 63 142H41Q38 142 38 139ZM74 128H210V139Q210 142 207 142H77Q74 142 74 139Z"
        fill={pcbFill} stroke={pcbStroke} strokeWidth="1.7" strokeLinejoin="round" />
      {GOLD_PINS.map((pin) => <rect key={pin} x={pin} y="131" width="4" height="10.5" rx=".9" fill={paint("gold")} />)}

      {/* Main rounded shroud */}
      <rect x="20" y="10" width="274" height="120" rx="12"
        fill={paint("shroud")} stroke={shroudStroke} strokeWidth="2.2" filter={paint("shadow")} />

      {/* Recessed top accent band + status LED bar */}
      <path d="M21 22Q21 11 32 11H282Q293 11 293 22H21Z" fill={topTrim} />
      <rect x="138" y="14.2" width="96" height="4.2" rx="2.1" fill={glow} opacity={0.48 + a * 0.36 + pulse * 0.16} />

      {/* Passive vent grille */}
      <rect x="267" y="29" width="19" height="86" rx="4" fill="#e1e6e2" stroke="#a8b0aa" strokeWidth="1.35" />
      <path d="M271 37h11m-11 7.8h11m-11 7.8h11m-11 7.8h11m-11 7.8h11m-11 7.8h11m-11 7.8h11m-11 7.8h11m-11 7.8h11m-11 7.8h11"
        stroke={finStroke} strokeWidth="1.55" strokeLinecap="round" />

      {/* Corner fasteners */}
      {[[31, 31], [31, 117], [257, 31], [257, 117]].map(([cx, cy]) => (
        <circle key={`${cx}/${cy}`} cx={cx} cy={cy} r="2.6" fill="#bfc5c0" stroke="#848c86" strokeWidth="1.05" />
      ))}

      {/* Dual counter-rotating fans */}
      {[86, 198].map((cx, i) => (
        <g key={cx} transform={`translate(${cx} 74)`}>
          <circle r="42.5" fill={rimFill} stroke={rimStroke} strokeWidth="1.55" />
          <circle r="38" fill={paint("fan")} />
          <g transform={`rotate(${i === 0 ? spin : -spin})`} data-gpu-rotor="">
            {BLADE_ANGLES.map((blade) => (
              <path key={blade}
                d="M3 -4.2C13 -9.2 27.5 -11.5 34.8 -3.6C36.1 -2 34.7 1.3 31.2 2.1C22 4 12.5 3 3 3.8Z"
                fill={paint("blade")} stroke={mix("#c5cdc8", "#b9cdc1", a)} strokeWidth=".7"
                strokeLinejoin="round" transform={`rotate(${blade})`} />
            ))}
          </g>
          <circle r="11.5" fill="#eff2ef" stroke={hubStroke} strokeWidth="1.55" />
          <circle r="5" fill={glow} opacity={0.65 + a * 0.35} />
          {a > 0.01 && <circle r="36.5" fill="none" stroke="#2aa95f" strokeWidth="1.9" opacity={a * (0.16 + pulse * 0.28)} />}
        </g>
      ))}

      {/* Center GPU badge */}
      <rect x="122" y="61" width="40" height="26" rx="6" fill="#fcfdfc" stroke={badgeStroke} strokeWidth="1.4" />
      <text x="142" y="78.5" textAnchor="middle" fill="#444d47" fontSize="12.5" fontWeight="650" letterSpacing="0.03em">GPU</text>

      {/* Smooth expanding intake ring centered on the left fan */}
      {p > 0 && p < 1 && (
        <circle cx="86" cy="74" r={16 + rippleEase * 22} fill="none" stroke="#2aa95f"
          strokeWidth={2.3 * (1 - p * 0.55)} opacity={Math.sin(p * Math.PI) * 0.44} />
      )}
    </g>
  );
}
