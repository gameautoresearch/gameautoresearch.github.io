import type { ComponentProps, ReactNode } from "react";
import { useAgentView } from "../../view";
import * as charts from "./charts";

// Figures, tables, quotes and asides, each with a plain-text variant for the agent view: charts become tables of
// their data, asides open inline, quotes become block quotes.

export type { Bar, Column, Series } from "./charts";

function TextTable({ head, rows }: { head: string[]; rows: ReactNode[][] }) {
  return (
    <div className="agent-table">
      <table>
        <thead>
          <tr>{head.map((h) => <th key={h}>{h}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((r, i) => <tr key={i}>{r.map((c, j) => <td key={j}>{c}</td>)}</tr>)}
        </tbody>
      </table>
    </div>
  );
}

export function Figure(props: ComponentProps<typeof charts.Figure>) {
  if (!useAgentView()) return <charts.Figure {...props} />;
  return (
    <figure className="agent-figure">
      <figcaption><strong>Figure {props.n}.</strong> {props.caption}</figcaption>
      {props.children}
    </figure>
  );
}

export function LineChart(props: ComponentProps<typeof charts.LineChart>) {
  if (!useAgentView()) return <charts.LineChart {...props} />;
  const { series, yFormat = String } = props;
  const many = series.length > 1;
  return (
    <TextTable
      head={[...(many ? ["Series"] : []), "Point", series.length === 1 ? series[0].name : "Value", "Note"]}
      rows={series.flatMap((s) => s.points.map((p) => [...(many ? [s.name] : []), p.label, yFormat(p.y), p.note ?? ""]))}
    />
  );
}

export function DeltaBars(props: ComponentProps<typeof charts.DeltaBars>) {
  if (!useAgentView()) return <charts.DeltaBars {...props} />;
  const { bars, unit = "Elo" } = props;
  const signed = (v: number) => (v > 0 ? `+${v}` : String(v));
  return (
    <TextTable
      head={["", unit, "95% interval", "Note"]}
      rows={bars.map((b) => [b.label, signed(b.value), b.lo !== undefined ? `${signed(b.lo)} to ${signed(b.hi!)}` : "", b.note])}
    />
  );
}

export function Columns(props: ComponentProps<typeof charts.Columns>) {
  if (!useAgentView()) return <charts.Columns {...props} />;
  const { cols, yFormat = String } = props;
  return (
    <TextTable
      head={["", "Value", "Note"]}
      rows={cols.map((c) => [c.label.replace(/\n/g, " "), c.display ?? yFormat(c.value), c.note])}
    />
  );
}

export function Table({ head, rows, align }: { head: string[]; rows: ReactNode[][]; align?: ("l" | "r")[] }) {
  if (useAgentView()) return <TextTable head={head} rows={rows} />;
  return (
    <div className="pt-table-wrap">
      <table className="pt-table">
        <thead>
          <tr>{head.map((h, i) => <th key={i} className={align?.[i] === "r" ? "r" : ""}>{h}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>{r.map((c, j) => <td key={j} className={align?.[j] === "r" ? "r" : ""}>{c}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function Aside({ title, children }: { title: string; children: ReactNode }) {
  if (useAgentView()) {
    return (
      <section className="agent-aside">
        <p><strong>Aside: {title}</strong></p>
        {children}
      </section>
    );
  }
  return (
    <details className="pt-aside">
      <summary>{title}</summary>
      <div className="pt-aside-body">{children}</div>
    </details>
  );
}

// A line from an agent's own transcript or notes, with who said it and where.
export function Quote({ children, who, where }: { children: ReactNode; who: string; where: ReactNode }) {
  return (
    <blockquote className="ar-quote">
      <p>{children}</p>
      <footer><strong>{who}</strong> · {where}</footer>
    </blockquote>
  );
}

export function Stats({ items }: { items: { value: string; label: string }[] }) {
  if (useAgentView()) {
    return <ul>{items.map((s) => <li key={s.label}>{s.value}: {s.label}</li>)}</ul>;
  }
  return (
    <div className="pt-stats">
      {items.map((s) => (
        <div key={s.label}><strong>{s.value}</strong><span>{s.label}</span></div>
      ))}
    </div>
  );
}

export function Cards({ items }: { items: { id?: string; kicker?: string; title: ReactNode; body: ReactNode }[] }) {
  if (useAgentView()) {
    return (
      <ul>
        {items.map((c) => (
          <li key={c.id ?? String(c.title)} id={c.id}>
            <strong>{c.kicker ? `${c.kicker}: ` : ""}{c.title}.</strong> {c.body}
          </li>
        ))}
      </ul>
    );
  }
  return (
    <div className="pt-cards">
      {items.map((c) => (
        <div key={c.id ?? String(c.title)} className="pt-card">
          {c.kicker && <span className="pt-card-kicker">{c.kicker}</span>}
          <h3 id={c.id}>{c.title}</h3>
          <p>{c.body}</p>
        </div>
      ))}
    </div>
  );
}
