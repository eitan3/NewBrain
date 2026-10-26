// Re-derives each module's inputs / outputs / prev / next from the edges of brain_spec.json, rewrites the JSON, and
// inlines it into index.html (between the BRAIN_SPEC markers) so the page works by double-click (no fetch over file://).
// Usage:  node embed_spec.mjs
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const specPath = join(here, "brain_spec.json"), htmlPath = join(here, "index.html");
const spec = JSON.parse(readFileSync(specPath, "utf8"));
const byId = Object.fromEntries(spec.nodes.map(n => [n.id, n]));
for (const e of spec.edges) {
  if (!byId[e.source] || !byId[e.target]) throw new Error(`edge ${e.id}: unknown module ${e.source} -> ${e.target}`);
}
for (const n of spec.nodes) {
  const ins = spec.edges.filter(e => e.target === n.id), outs = spec.edges.filter(e => e.source === n.id);
  const io = (e, other, key) => ({ [key]: other, name: byId[other].name, label: e.label, kind: e.kind, shape_sym: e.shape_sym, shape_num: e.shape_num });
  n.inputs = ins.map(e => io(e, e.source, "from"));
  n.outputs = outs.map(e => io(e, e.target, "to"));
  n.prev = [...new Set(ins.map(e => e.source))].sort();
  n.next = [...new Set(outs.map(e => e.target))].sort();
}
writeFileSync(specPath, JSON.stringify(spec, null, 1) + "\n");
const json = JSON.stringify(spec).replace(/</g, "\\u003c");   // safe inside <script>
const html = readFileSync(htmlPath, "utf8");
const re = /(<!-- BRAIN_SPEC_START[^>]*-->\s*<script type="application\/json" id="brain-spec">)[\s\S]*?(<\/script>\s*<!-- BRAIN_SPEC_END -->)/;
if (!re.test(html)) throw new Error("BRAIN_SPEC markers not found in index.html");
writeFileSync(htmlPath, html.replace(re, (_, a, b) => a + json + b));
console.log(`embedded ${spec.nodes.length} modules, ${spec.edges.length} edges (${(json.length / 1024).toFixed(0)} KB)`);
