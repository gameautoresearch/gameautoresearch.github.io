import { useId } from "react";

const BLADE_ANGLES = Array.from({ length: 9 }, (_, i) => i * 40);
const GOLD_PINS = [...[42, 48, 54, 60], ...Array.from({ length: 18 }, (_, i) => 79 + i * 7)];

export default function GpuCard({ x, y, active, t, intake }: {
  x: number; y: number; active: boolean; t: number; intake: number;
}) {
  const id = useId();
  const paint = (name: string) => `url(#${id}-${name})`;
  const angle = (t * (active ? 0.82 : 0.2)) % 360;
  const glow = active ? "#25b562" : "#7e8480";
  return (
    <g transform={`translate(${x} ${y + intake}) scale(.415)`}>
      <defs>
        <linearGradient id={`${id}-shroud`} x2="0" y2="100%">
          <stop stopColor={active ? "#f1faf4" : "#f5f7f5"} />
          <stop offset=".55" stopColor={active ? "#e3f2e8" : "#eaece9"} />
          <stop offset="1" stopColor={active ? "#d3e7da" : "#dcdfd9"} />
        </linearGradient>
        <linearGradient id={`${id}-fan`} x2="0" y2="100%">
          <stop stopColor="#2f3431" />
          <stop offset="1" stopColor="#464c48" />
        </linearGradient>
        <linearGradient id={`${id}-blade`} x2="0" y2="100%">
          <stop stopColor="#818984" />
          <stop offset="1" stopColor="#454c47" />
        </linearGradient>
        <linearGradient id={`${id}-gold`} x2="0" y2="100%">
          <stop stopColor="#dfc686" />
          <stop offset="1" stopColor="#be9e53" />
        </linearGradient>
        <filter id={`${id}-shadow`} x="-15%" y="-30%" width="130%" height="160%">
          <feDropShadow dx="0" dy="7" stdDeviation="7" floodColor="#152019" floodOpacity=".2" />
        </filter>
      </defs>
      <rect x="214" y="2" width="32" height="12" rx="2" fill="#3d413e" />
      {[220, 226, 232, 238].map((pin) => <circle key={pin} cx={pin} cy="7" r="1.5" fill="#d4b670" />)}
      <path d="M76 12C76 4 98 4 98 12M108 12C108 4 130 4 130 12" fill="none" stroke="#c08c63" strokeWidth="3.5" strokeLinecap="round" />
      <path d="M-2 0H20V142H10V10H-2Z" fill="#d7dad6" stroke="#434643" strokeWidth="2.2" strokeLinejoin="round" />
      <path d="M38 128H66V142H38ZM74 128H210V142H74Z" fill="#385042" stroke="#434643" strokeWidth="1.8" />
      {GOLD_PINS.map((pin) => <rect key={pin} x={pin} y="131" width="4" height="11" rx=".7" fill={paint("gold")} />)}
      <path d="M20 18L28 10H282L294 20V120L284 130H28L20 122Z" fill={paint("shroud")} stroke={active ? "#2f9e5c" : "#434643"} strokeWidth="2.4" strokeLinejoin="round" filter={paint("shadow")} />
      <path d="M28 11H281L292 21H21Z" fill="#363a37" />
      <rect x="138" y="14" width="96" height="4.5" rx="2.2" fill={glow} opacity={active ? .74 + intake * .26 : .45} />
      <rect x="267" y="28" width="19" height="88" rx="3" fill="#d4d8d4" stroke="#8d938e" strokeWidth="1.4" />
      <path d="M270 36h13m-13 8h13m-13 8h13m-13 8h13m-13 8h13m-13 8h13m-13 8h13m-13 8h13m-13 8h13m-13 8h13" stroke="#6e746f" strokeWidth="1.7" />
      {[[30, 29], [30, 117], [258, 29], [258, 117]].map(([cx, cy]) => (
        <circle key={`${cx}/${cy}`} cx={cx} cy={cy} r="3.1" fill="#a3a8a3" stroke="#5c615d" />
      ))}
      {[86, 198].map((cx, i) => (
        <g key={cx} transform={`translate(${cx} 74)`}>
          <circle r="42.5" fill="#cfd4cf" stroke="#7c827d" strokeWidth="1.6" />
          <circle r="38" fill={paint("fan")} />
          <g transform={`rotate(${i === 0 ? angle : -angle})`} data-gpu-rotor="">
            {BLADE_ANGLES.map((blade) => (
              <path key={blade} d="M2 -5C13 -8 30 -16 36 -4C28 2 14 2 2 4Z" fill={paint("blade")} stroke="#858e88" strokeWidth=".7" transform={`rotate(${blade})`} />
            ))}
          </g>
          <circle r="11.5" fill="#e7eae6" stroke="#4e544f" strokeWidth="1.6" />
          <circle r="5.5" fill={glow} />
          {active && <circle r="36" fill="none" stroke={glow} strokeWidth="2.3" opacity={.16 + intake * .5} />}
        </g>
      ))}
      <rect x="122" y="60" width="40" height="28" rx="4" fill="#fff" stroke="#b5bcb6" />
      <text x="142" y="79" textAnchor="middle" fill="#373e39" fontSize="13" fontWeight="750">GPU</text>
      {intake > 0 && <circle cx="86" cy="48" r={12 + (1 - intake) * 26} fill="none" stroke={glow} strokeWidth={3 * intake} opacity={intake * .8} />}
    </g>
  );
}
