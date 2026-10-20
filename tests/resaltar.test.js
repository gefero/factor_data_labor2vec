// Prueba de la lógica de resaltado del mapa 3D con un entorno simulado (sin navegador).
// Uso: node tests/resaltar.test.js ruta/al/resaltar.js
const assert = require('assert');
const fs = require('fs');
const creados = [];
function el(tag) { const e = { tag, style: {}, children: [], appendChild(c) { this.children.push(c); return c; },
  parentNode: null, insertBefore() {}, set textContent(v) { this._t = v; }, get textContent() { return this._t; } };
  creados.push(e); return e; }
const gd = el('div');
gd.data = [
  {name: 'Rama: Agricultura', meta: {grupo: 'Ramas', base: 3.5}},
  {name: 'Rama: Pesca', meta: {grupo: 'Ramas', base: 3.5}},
  {name: 'Provincia: NOA', meta: {grupo: 'Provincia', base: 6}},
  {name: 'Salario', meta: {grupo: 'Salario', base: 4}},
  {name: 'Sexo', meta: {grupo: 'Sexo', base: 7}},
];
gd.parentNode = el('div');
const handlers = {}; gd.on = (ev, fn) => { handlers[ev] = fn; };
const llamadas = [];
global.Plotly = { restyle: (g, attrs, idx) => llamadas.push({attrs: JSON.parse(JSON.stringify(attrs)), idx}) };
global.document = { getElementById: () => gd, createElement: t => el(t), createTextNode: t => ({t}) };
eval(fs.readFileSync(process.argv[2], 'utf8'));
const ult = () => llamadas[llamadas.length - 1].attrs;
const boton = t => creados.find(e => e.tag === 'button' && e.textContent === t);

assert.deepStrictEqual(ult().opacity, [1, 1, 1, 1, 1]);
assert.deepStrictEqual(ult()['marker.size'], [3.5, 3.5, 6, 4, 7]);
// clic en la leyenda: resalta una traza, atenúa el resto, y no apaga el elemento (return false)
assert.strictEqual(handlers['plotly_legendclick']({curveNumber: 0}), false);
assert.deepStrictEqual(ult().opacity, [1, 0.05, 0.05, 0.05, 0.05]);
assert.deepStrictEqual(ult().visible, [true, true, true, true, true]);
assert.strictEqual(ult()['marker.size'][0], 3.5 * 1.6);
assert.strictEqual(ult()['marker.size'][1], 3.5);
// selección múltiple y quitar de a una
handlers['plotly_legendclick']({curveNumber: 2});
assert.deepStrictEqual(ult().opacity, [1, 0.05, 1, 0.05, 0.05]);
handlers['plotly_legendclick']({curveNumber: 0});
assert.deepStrictEqual(ult().opacity, [0.05, 0.05, 1, 0.05, 0.05]);
handlers['plotly_legendclick']({curveNumber: 2});
assert.deepStrictEqual(ult().opacity, [1, 1, 1, 1, 1]);
assert.strictEqual(handlers['plotly_legenddoubleclick'](), false);
// botón de categoría: selecciona todas las trazas del grupo "Ramas"
boton('Ramas (todos los sectores)').onclick();
assert.deepStrictEqual(ult().opacity, [1, 1, 0.05, 0.05, 0.05]);
assert.ok(/Resaltado: Rama: Agricultura; Rama: Pesca/.test(creados.find(e => e.tag === 'span').textContent));
// casilla "Ocultar lo no resaltado": lo no resaltado deja de ser visible
const cb = creados.find(e => e.tag === 'input');
cb.checked = true; cb.onchange();
assert.deepStrictEqual(ult().visible, [true, true, false, false, false]);
// quitar el resaltado: todo vuelve, aun con la casilla marcada
boton('Quitar resaltado').onclick();
assert.deepStrictEqual(ult().visible, [true, true, true, true, true]);
assert.deepStrictEqual(ult().opacity, [1, 1, 1, 1, 1]);
console.log('JS de resaltado: OK');
