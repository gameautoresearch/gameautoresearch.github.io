// Turn scrubber: one stop per turn. The rail, the filled part, the knockout dots and the thumb all sit on one
// centre line and share one coordinate (value / max of the inner width), so a dot at turn t is exactly under the
// thumb at turn t.
import { useRef, type KeyboardEvent, type PointerEvent } from "react";

export interface ScrubberMark {
  at: number;
  className?: string;
  title?: string;
}

interface Props {
  value: number;
  max: number;
  onSeek: (value: number) => void;
  marks?: ScrubberMark[];
  label: string;
  valueText: string;
}

export function Scrubber({ value, max, onSeek, marks = [], label, valueText }: Props) {
  const inner = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);
  const pct = (v: number) => `${(Math.min(max, Math.max(0, v)) / max) * 100}%`;

  const fromPointer = (e: PointerEvent) => {
    const r = inner.current!.getBoundingClientRect();
    const v = Math.round(((e.clientX - r.left) / r.width) * max);
    return Math.min(max, Math.max(0, v));
  };
  const seek = (v: number) => { if (v !== value) onSeek(v); };

  const onPointerDown = (e: PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    dragging.current = true;
    e.currentTarget.setPointerCapture(e.pointerId);
    e.currentTarget.focus();
    seek(fromPointer(e));
  };
  const onPointerMove = (e: PointerEvent<HTMLDivElement>) => { if (dragging.current) seek(fromPointer(e)); };
  const onPointerUp = () => { dragging.current = false; };
  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    const to = { ArrowLeft: value - 1, ArrowDown: value - 1, ArrowRight: value + 1, ArrowUp: value + 1, Home: 0, End: max }[e.key];
    if (to === undefined) return;
    e.preventDefault();
    e.stopPropagation();
    seek(Math.min(max, Math.max(0, to)));
  };

  return (
    <div className="psr-scrub" role="slider" tabIndex={0} aria-label={label} aria-valuemin={0} aria-valuemax={max}
      aria-valuenow={value} aria-valuetext={valueText}
      onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp} onPointerCancel={onPointerUp}
      onKeyDown={onKeyDown}>
      <div className="psr-scrub-inner" ref={inner}>
        <div className="psr-rail" />
        <div className="psr-fill" style={{ width: pct(value) }} />
        {marks.map((m, i) => <i key={i} className={`psr-dot ${m.className ?? ""}`} style={{ left: pct(m.at) }} title={m.title} />)}
        <div className="psr-thumb" style={{ left: pct(value) }} />
      </div>
    </div>
  );
}
