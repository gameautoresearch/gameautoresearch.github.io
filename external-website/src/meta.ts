// The site is a single post. vite.config.ts writes these link-preview tags into index.html, because crawlers don't
// run the app.
export const PAGE = {
  title:
    "Game Autoresearch Bench: Can AI agents train strong game bots on a budget?",
  date: "2026-11-25",
  authors: [
    { name: "Kevin Zhang", url: "https://x.com/kzdagoof" },
    { name: "Max Bittker", url: "https://x.com/maxbittker" },
  ],
  description:
    "We gave AI agents a GPU budget and had them train reinforcement learning bots for Pokémon, Polytopia and Generals: how well they do it, which models are best, and what separates good research from bad.",
  ogImage: "/og.png",
  ogImageAlt:
    "Game Autoresearch Bench: AI agents given a $14.50 NVIDIA L40S GPU budget iterate on reinforcement learning training runs and evaluate policies across Polytopia, Pokémon 3v3, and Generals.io.",
};

const DEFAULT_SITE_URL = "https://gameautoresearch.github.io";

const escape = (s: string) =>
  s
    .replace(/&/g, "&amp;")
    .replace(/"/g, "&quot;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

export function metaTags(siteUrl: string) {
  const { title, description, ogImage, ogImageAlt } = PAGE;
  const imageUrl = `${siteUrl || DEFAULT_SITE_URL}${ogImage}`;
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
    `<meta property="og:image" content="${imageUrl}" />`,
    `<meta property="og:image:type" content="image/png" />`,
    `<meta property="og:image:width" content="1200" />`,
    `<meta property="og:image:height" content="630" />`,
    `<meta property="og:image:alt" content="${escape(ogImageAlt)}" />`,
    `<meta name="twitter:card" content="summary_large_image" />`,
    `<meta name="twitter:title" content="${escape(title)}" />`,
    `<meta name="twitter:description" content="${escape(description)}" />`,
    `<meta name="twitter:image" content="${imageUrl}" />`,
    `<meta name="twitter:image:alt" content="${escape(ogImageAlt)}" />`,
  ].join("\n    ");
}
