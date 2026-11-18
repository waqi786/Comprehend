const fs = require('fs');
const dag = require('d3-dag');

const htmlPath = 'visualisation/viewer_512_base_standalone.html';
const html = fs.readFileSync(htmlPath, 'utf8');

const m = html.match(/window\.__EMBEDDED_TREE_DATA__\s*=\s*(\[[\s\S]*?\]);/);
if (!m) { console.error('Data not found'); process.exit(1); }
const data = JSON.parse(m[1]);
console.log(`Loaded ${data.length} nodes`);

// Compute layer (depth) manually — same as browser's n.layer
const byId = {};
data.forEach(d => { byId[String(d.id)] = d; });
const layerOf = {};
function getLayer(d) {
  const id = String(d.id);
  if (id in layerOf) return layerOf[id];
  const parents = d.parentIds || [];
  if (parents.length === 0) { layerOf[id] = 0; return 0; }
  let maxP = 0;
  parents.forEach(p => {
    const parent = byId[String(p)];
    if (parent) {
      const pl = getLayer(parent);
      if (pl > maxP) maxP = pl;
    }
  });
  layerOf[id] = maxP + 1;
  return layerOf[id];
}
data.forEach(d => getLayer(d));
const bottomLayer = Math.max(...Object.values(layerOf));
console.log(`bottomLayer = ${bottomLayer}`);
data.forEach(d => { d.layer = layerOf[String(d.id)]; });

const root = dag.dagStratify().id(d => d.id).parentIds(d => d.parentIds || [])(data);

// EXACT same params as browser
const BASE = 200, V_GAP = 120;
const W = 1240, H = 540;  // typical browser viewport - margins

const layout = dag.sugiyama()
  .size([W, H])
  .layering(dag.layeringSimplex())
  .decross(dag.decrossTwoLayer())
  .nodeSize(n => {
    const l = bottomLayer - (n.data.layer || 0);
    return [l === 0 ? BASE : BASE * Math.pow(2, l), V_GAP];
  })
  .coord(dag.coordCenter());

layout(root);
console.log('Layout done');

// Bounds check
let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
root.descendants().forEach(n => {
  if (n.x < minX) minX = n.x;
  if (n.x > maxX) maxX = n.x;
  if (n.y < minY) minY = n.y;
  if (n.y > maxY) maxY = n.y;
});
console.log(`Bounds X: [${minX}, ${maxX}], Y: [${minY}, ${maxY}]`);

const positions = {};
let nan_count = 0;
root.descendants().forEach(n => {
  if (isNaN(n.x) || isNaN(n.y)) nan_count++;
  positions[String(n.data.id)] = { x: n.x, y: n.y };
});
console.log(`NaN count: ${nan_count}`);

let count = 0;
data.forEach(d => {
  const p = positions[String(d.id)];
  if (p && !isNaN(p.x) && !isNaN(p.y)) { d.x = p.x; d.y = p.y; count++; }
});
console.log(`Positions set for ${count} nodes`);

// Replace JSON
const newJson = JSON.stringify(data);
const dataStart = m.index + m[0].indexOf('[');
const dataEnd = m.index + m[0].lastIndexOf(']') + 1;
let newHtml = html.substring(0, dataStart) + newJson + html.substring(dataEnd);

// Patch init()
const oldInit = `    root = stratify(data);
    const layout = d3.sugiyama().size([container.clientWidth - margin.left - margin.right, container.clientHeight - margin.top - margin.bottom]);
    layout(root);
    update();`;
const newInit = `    root = stratify(data);
    root.descendants().forEach(n => {
      n.x = (n.data && n.data.x !== undefined) ? n.data.x : 0;
      n.y = (n.data && n.data.y !== undefined) ? n.data.y : 0;
    });
    update();`;
newHtml = newHtml.replace(oldInit, newInit);

// Patch update()
const oldUpd = `  const bottomLayer = d3.max(root.descendants(), n => n.layer);
  const layout = d3.sugiyama().size([w, h]).layering(d3.layeringSimplex()).decross(d3.decrossTwoLayer())
    .nodeSize(n => { const l = bottomLayer - n.layer; return [l === 0 ? BASE : BASE * Math.pow(2, l), V_GAP]; })
    .coord(d3.coordCenter());
  layout(root);`;
const newUpd = `  // Precomputed positions (scaled to match browser viewport)`;
newHtml = newHtml.replace(oldUpd, newUpd);

fs.writeFileSync('visualisation/viewer_512_base_fast.html', newHtml);
console.log('Wrote viewer_512_base_fast.html');
