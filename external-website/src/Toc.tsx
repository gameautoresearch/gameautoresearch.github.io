import { useEffect, useState } from "react";
import "./toc.css";

type Entry = { id: string; label: string; level: 2 | 3; plate?: string };

// The post's table of contents, pinned in the left margin on wide screens. It is read from the rendered headings
// (every h2 and h3 with an id), so it stays in step with the post without a second list to maintain. The
// configuration headings carry a <small> kicker; the nav shows only their title.
function readHeadings(root: HTMLElement): Entry[] {
  return Array.from(root.querySelectorAll<HTMLHeadingElement>("h2[id], h3[id]")).map((h) => {
    const clone = h.cloneNode(true) as HTMLElement;
    clone.querySelector("small")?.remove();
    const plate = ["ar-c1", "ar-c2", "ar-c3"].find((c) => h.classList.contains(c));
    return { id: h.id, label: clone.textContent?.trim() ?? h.id, level: h.tagName === "H2" ? 2 : 3, plate };
  });
}

export default function Toc({ rootId }: { rootId: string }) {
  const [entries, setEntries] = useState<Entry[]>([]);
  const [active, setActive] = useState<string | null>(null);

  useEffect(() => {
    const root = document.getElementById(rootId);
    if (!root) return;
    let found = readHeadings(root);
    setEntries(found);

    // Re-read when the post's headings change (sections that render late, or edits during development).
    const mo = new MutationObserver(() => {
      const next = readHeadings(root);
      if (next.map((e) => e.id + e.label).join() === found.map((e) => e.id + e.label).join()) return;
      found = next;
      setEntries(next);
      onScroll();
    });
    mo.observe(root, { childList: true, subtree: true });

    // Scroll spy: the active section is the last heading whose top has passed a line near the top of the window.
    let frame = 0;
    const update = () => {
      frame = 0;
      const line = 120;
      let current: string | null = null;
      for (const e of found) {
        const el = document.getElementById(e.id);
        if (el && el.getBoundingClientRect().top <= line) current = e.id;
      }
      setActive(current);
    };
    const onScroll = () => {
      if (!frame) frame = requestAnimationFrame(update);
    };
    update();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => {
      mo.disconnect();
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
      if (frame) cancelAnimationFrame(frame);
    };
  }, [rootId]);

  if (!entries.length) return null;

  const go = (id: string) => (ev: React.MouseEvent) => {
    const el = document.getElementById(id);
    if (!el) return;
    ev.preventDefault();
    el.scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
    window.history.replaceState(window.history.state, "", `#${id}`);
  };

  return (
    <nav className="toc" aria-label="Contents">
      <p className="toc-title">Contents</p>
      <ol>
        {entries.map((e, i) => (
          <li
            key={e.id}
            // Sections before the first configuration heading sit at the top level, not indented under one.
            className={`toc-l${e.level === 3 && !entries.slice(0, i).some((p) => p.level === 2) ? "0" : e.level}${e.plate ? ` toc-${e.plate}` : ""}`}
          >
            <a href={`#${e.id}`} onClick={go(e.id)} aria-current={active === e.id ? "location" : undefined}>
              {e.label}
            </a>
          </li>
        ))}
      </ol>
    </nav>
  );
}
