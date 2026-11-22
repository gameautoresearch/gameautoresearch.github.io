// The site is a single post. vite.config.ts writes these link-preview tags into index.html, because crawlers don't
// run the app.
export const PAGE = {
  title:
    "VibeResearch Bench: Can AI agents train strong game bots on a budget?",
  date: "2026-11-19",
  authors: [
    { name: "Kevin Zhang", url: "https://x.com/kzdagoof" },
    { name: "Max Bittker", url: "https://x.com/maxbittker" },
  ],
  description:
    "We gave AI agents a GPU budget and had them train reinforcement learning bots for Pokémon, Polytopia and Generals: how well they do it, which models are best, and what separates good research from bad.",
};

const escape = (s: string) =>
  s
    .replace(/&/g, "&amp;")
    .replace(/"/g, "&quot;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

export function metaTags(siteUrl: string) {
  const { title, description } = PAGE;
  return [
    `<meta name="description" content="${escape(description)}" />`,
    ...(siteUrl
      ? [
          `<link rel="canonical" href="${siteUrl}/" />`,
          `<meta property="og:url" content="${siteUrl}/" />`,
        ]
      : []),
    `<meta property="og:type" content="article" />`,
    `<meta property="og:title" content="${escape(title)}" />`,
    `<meta property="og:description" content="${escape(description)}" />`,
    `<meta name="twitter:card" content="summary" />`,
    `<meta name="twitter:title" content="${escape(title)}" />`,
    `<meta name="twitter:description" content="${escape(description)}" />`,
  ].join("\n    ");
}
