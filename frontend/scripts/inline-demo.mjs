// Bundle the demo build into one self-contained HTML fragment (JS + CSS inlined),
// suitable for hosting as a static page or a claude.ai artifact.
import { readFileSync, writeFileSync, readdirSync } from "node:fs";
import { join } from "node:path";

const dir = "dist-demo";
const assets = readdirSync(join(dir, "assets"));
const js = assets.find((f) => f.endsWith(".js"));
const css = assets.find((f) => f.endsWith(".css"));
const read = (f) => readFileSync(join(dir, "assets", f), "utf8");

const html = `<title>EvenementsRAG Demo</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover" />
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;700&display=swap" />
<style>${read(css)}</style>
<style>:root{color-scheme:light}body{background:#f9fafb;color:#111827}</style>
<div id="root"></div>
<script type="module">${read(js).replace(/<\/script/g, "<\\/script")}</script>
`;
writeFileSync(join(dir, "evenementsrag-demo.html"), html);
console.log("wrote", join(dir, "evenementsrag-demo.html"), (html.length / 1024).toFixed(0), "KB");
