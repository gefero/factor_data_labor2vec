"""Figuras del espacio de conceptos: mapa PaCMAP, dispersión entre modelos e histograma de hubness."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pacmap  # noqa: E402

from labor2vec.analisis.metricas import distancias_coseno  # noqa: E402

NOMBRES_CATEGORIA = {"sexo": "Sexo", "nacimiento": "Año de nacimiento", "salario": "Salario (percentil)",
                     "tamaño": "Tamaño de empresa", "rama": "Rama (r34)", "rama2": "Rama (r32)",
                     "provincia": "Provincia", "antigüedad": "Antigüedad de la empresa"}


def proyectar(e: np.ndarray, n_componentes: int = 2) -> np.ndarray:
    """PaCMAP con los parámetros del notebook concept_space.ipynb de life2vec (2D por defecto)."""
    return pacmap.PaCMAP(n_components=n_componentes, n_neighbors=None, random_state=0, MN_ratio=1, FP_ratio=10,
                         distance="angular", lr=0.5).fit_transform(e.astype("float32"))


def _guardar(fig, ruta):
    fig.tight_layout()
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


PREFIJO_CATEGORIA = {"tamaño": "Tamaño", "antigüedad": "Antigüedad", "sexo": "Sexo"}


def mapa_categorias(xy, tabla, ruta, titulo: str) -> None:
    """Mapa de todos los conceptos por categoría. Tamaño de empresa, antigüedad y sexo (pocos tokens) se dibujan al final,
    más grandes y con su nombre."""
    cats = list(NOMBRES_CATEGORIA)
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(9, 8))
    chicas = list(PREFIJO_CATEGORIA)
    cat = tabla["categoria"].to_numpy()
    for i, c in enumerate(cats):
        if c in chicas:
            continue
        m = cat == c
        if m.any():
            ax.scatter(xy[m, 0], xy[m, 1], s=22, color=cmap(i), label=f"{NOMBRES_CATEGORIA[c]} ({m.sum()})",
                       edgecolors="white", linewidths=0.4, alpha=0.9)
    amplitud = max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]))
    desplazamientos = {"tamaño": (-150, -90), "antigüedad": (90, -110), "sexo": (110, 70)}
    for c in chicas:
        i = cats.index(c)
        m = np.flatnonzero(cat == c)
        ax.scatter(xy[m, 0], xy[m, 1], s=70, color=cmap(i), marker="s", label=f"{NOMBRES_CATEGORIA[c]} ({len(m)})",
                   edgecolors="black", linewidths=0.9, zorder=3)
        orden = m[np.argsort(tabla["valor"].to_numpy()[m])]
        nombres = [str(tabla["descripcion"][int(k)]) for k in orden]
        centro = xy[m].mean(0)
        # los tokens de estas categorías suelen caer casi en el mismo punto: una sola nota con flecha lista todos los nombres
        dispersion = np.max(np.linalg.norm(xy[m][:, None] - xy[m][None], axis=2)) / amplitud
        if dispersion < 0.08:
            texto = f"{PREFIJO_CATEGORIA[c]} ({len(m)} puntos casi superpuestos):\n" + "\n".join(f"· {n_}" for n_ in nombres)
            ax.annotate(texto, centro, xytext=desplazamientos[c], textcoords="offset points", fontsize=8, va="center",
                        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=cmap(i), linewidth=1.2),
                        arrowprops=dict(arrowstyle="-", color=cmap(i), linewidth=1.2), zorder=4)
        else:
            for k, n_ in zip(orden, nombres):
                ax.annotate(f"{PREFIJO_CATEGORIA[c]}: {n_}", xy[k], xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_title(titulo)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=8, loc="lower left", frameon=False)
    _guardar(fig, ruta)


def paneles(xy, tabla, ruta, titulo: str) -> None:
    cat = tabla["categoria"].to_numpy()
    fig, axs = plt.subplots(2, 2, figsize=(11, 9.5))
    for ax in axs.ravel():
        ax.scatter(xy[:, 0], xy[:, 1], s=6, color="#cccccc")
        ax.set_xticks([])
        ax.set_yticks([])
    for ax, c, nombre in ((axs[0, 0], "salario", "Salario: percentil"), (axs[0, 1], "nacimiento", "Año de nacimiento")):
        m = cat == c
        sc = ax.scatter(xy[m, 0], xy[m, 1], s=30, c=tabla["valor"].to_numpy()[m].astype(float), cmap="viridis",
                        edgecolors="white", linewidths=0.4)
        fig.colorbar(sc, ax=ax, shrink=0.8)
        ax.set_title(nombre)
    ax = axs[1, 0]
    m = cat == "provincia"
    regiones = sorted(set(tabla["region"].to_numpy()[m]))
    cmap = plt.get_cmap("tab10")
    for i, r in enumerate(regiones):
        mm = m & (tabla["region"].to_numpy() == r)
        ax.scatter(xy[mm, 0], xy[mm, 1], s=50, color=cmap(i), label=r, edgecolors="white", linewidths=0.5)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Provincia: región")
    ax = axs[1, 1]
    m = cat == "rama"
    letras = tabla["letra"].to_numpy()
    sc = ax.scatter(xy[m, 0], xy[m, 1], s=18, c=letras[m].astype(float), cmap="tab20", edgecolors="white",
                    linewidths=0.3)
    fig.colorbar(sc, ax=ax, shrink=0.8, label="letra (sector)")
    ax.set_title("Rama (r34): letra del sector")
    fig.suptitle(titulo)
    _guardar(fig, ruta)


def dispersion_modelos(distancias: dict, pares: list[tuple[str, str]], ruta, n_pares: int = 10000, rng=None) -> None:
    """Distancia coseno de pares de tokens al azar en un modelo contra la del mismo par en otro, como en el original."""
    rng = rng or np.random.default_rng(0)
    n = next(iter(distancias.values())).shape[0]
    i, j = np.triu_indices(n, 1)
    sel = rng.choice(len(i), size=min(n_pares, len(i)), replace=False)
    fig, axs = plt.subplots(1, len(pares), figsize=(4.2 * len(pares), 4.2), squeeze=False)
    for ax, (a, b) in zip(axs[0], pares):
        ax.scatter(distancias[a][i[sel], j[sel]], distancias[b][i[sel], j[sel]], s=2, alpha=0.2, color="#3b6ea5")
        ax.set_xlabel(f"distancia coseno ({a})")
        ax.set_ylabel(f"distancia coseno ({b})")
        ax.set_title(f"{a} contra {b}")
    _guardar(fig, ruta)


def hubness(grados: dict, ruta, k: int) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    maxg = max(int(g.max()) for g in grados.values())
    bins = np.arange(0, maxg + 2) - 0.5
    for nombre, g in grados.items():
        ax.hist(g, bins=bins, alpha=0.5, label=f"{nombre} (mediana {np.median(g):.0f})")
    ax.axvline(k, color="k", linestyle="--", linewidth=0.8)
    ax.set_xlabel(f"grado de entrada en el grafo de {k} vecinos")
    ax.set_ylabel("tokens")
    ax.set_title("Hubness: la línea punteada marca el grado medio")
    ax.legend(fontsize=8, frameon=False)
    _guardar(fig, ruta)


def distancias(e: np.ndarray) -> np.ndarray:
    return distancias_coseno(e)


# ---- ramas de actividad con nombres ---------------------------------------------------------------------------
NOMBRE_LETRA = {1: "Agricultura y ganadería", 2: "Pesca", 3: "Minería", 4: "Industria manufacturera",
                5: "Electricidad, gas y agua", 6: "Construcción", 7: "Comercio", 8: "Hotelería y restaurantes",
                9: "Transporte y comunicaciones", 10: "Finanzas", 11: "Servicios inmobiliarios y empresariales",
                12: "Enseñanza", 13: "Salud y servicios sociales", 14: "Otros servicios comunitarios y personales"}
COLOR_LETRA = {i: c for i, c in zip(range(1, 15), [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#008080", "#9a6324",
    "#800000", "#808000", "#000075", "#808080"])}


def _ramas(tabla):
    """Filas de la tabla de conceptos que son ramas (r34 y r32 agregadas): máscara y arreglos auxiliares."""
    cat = tabla["categoria"].to_numpy()
    return np.isin(cat, ["rama", "rama2"]), cat


def corto(descripcion: str, n: int = 34) -> str:
    """Nombre abreviado de una división: hasta la primera coma o punto y coma, recortado a `n` caracteres."""
    base = descripcion.replace("\n", " ").split(";")[0].split(",")[0].strip()
    return base if len(base) <= n else base[: n - 1].rstrip() + "…"


def _etiquetar_sin_solapar(fig, ax, items: list[tuple[float, float, str, str]], fontsize: float = 7.5) -> int:
    """Pone cada etiqueta (x, y, texto, color) al lado de su punto probando cuatro posiciones; si ninguna entra sin pisar
    una etiqueta ya puesta, se omite. Los items van en orden de prioridad. Devuelve cuántas se pusieron."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    puestos = []
    opciones = [(5, 4, "left", "bottom"), (5, -4, "left", "top"), (-5, 4, "right", "bottom"), (-5, -4, "right", "top")]
    n = 0
    for x, y, texto, color in items:
        for dx, dy, ha, va in opciones:
            t = ax.annotate(texto, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=fontsize, ha=ha,
                            va=va, color=color)
            bb = t.get_window_extent(renderer).expanded(1.05, 1.15)
            if not any(bb.overlaps(o) for o in puestos):
                puestos.append(bb)
                n += 1
                break
            t.remove()
    return n


def mapa_ramas(xy, tabla, ruta, titulo: str, n_nombres: int = 40) -> None:
    """Mapa de las ramas coloreado por sector (panel izquierdo) y con el nombre de las `n_nombres` ramas más frecuentes,
    sin solaparse (panel derecho). No se rotula el centro de cada sector: los sectores no forman grupos compactos."""
    m, cat = _ramas(tabla)
    letra = tabla["letra"].to_numpy()
    fig, axs = plt.subplots(1, 2, figsize=(17, 8.2))
    for ax in axs:
        for l in sorted(NOMBRE_LETRA):
            mm = m & (letra == l) & (cat == "rama")
            ax.scatter(xy[mm, 0], xy[mm, 1], s=30, color=COLOR_LETRA[l], edgecolors="white", linewidths=0.4,
                       label=NOMBRE_LETRA[l] if ax is axs[0] else None)
        m2 = m & (cat == "rama2")
        ax.scatter(xy[m2, 0], xy[m2, 1], s=60, marker="s", color="none", edgecolors="black", linewidths=1.0,
                   label="Rama agregada a 2 dígitos (r32)" if ax is axs[0] else None)
        ax.set_xticks([])
        ax.set_yticks([])
    axs[0].set_title("Color: sector de la rama (los sectores no forman grupos compactos)")
    axs[0].legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3, frameon=False)
    frec = tabla["frecuencia"].to_numpy()
    desc = tabla["descripcion"].to_numpy()
    token = tabla["token"].to_numpy()
    orden = [i for i in np.argsort(-frec) if m[i]][:n_nombres]
    items = []
    for i in orden:
        nombre = corto(str(desc[i]) or str(token[i]), 30)
        items.append((xy[i, 0], xy[i, 1], nombre, "#222222"))
    n = _etiquetar_sin_solapar(fig, axs[1], items)
    axs[1].set_title(f"Nombre de las {n} ramas más frecuentes (de las {n_nombres} con más eventos)")
    fig.suptitle(titulo)
    _guardar(fig, ruta)


def html_ramas(xy, tabla, ruta) -> None:
    """Mapa interactivo autocontenido (SVG): al pasar el cursor por un punto aparece el token, su descripción, el sector y
    la división. Los cuadrados son las ramas agregadas a 2 dígitos."""
    m, cat = _ramas(tabla)
    x, y = xy[m, 0], xy[m, 1]
    w, h, mg = 900, 720, 30
    px = mg + (x - x.min()) / (x.max() - x.min()) * (w - 2 * mg)
    py = mg + (1 - (y - y.min()) / (y.max() - y.min())) * (h - 2 * mg)
    puntos = []
    sub = tabla.filter(m) if hasattr(tabla, "filter") else None
    filas = list(zip(sub["token"].to_list(), sub["categoria"].to_list(), sub["letra"].to_list(), sub["r32"].to_list(),
                     sub["descripcion"].to_list(), sub["frecuencia"].to_list()))
    for (tok, c_, l, v32, desc, frec), a, b in zip(filas, px, py):
        color = COLOR_LETRA.get(int(l), "#999")
        info = f"{tok} — {desc or 'sin descripción'} | sector: {NOMBRE_LETRA.get(int(l), '?')} | división r32: {v32} | {frec:,} eventos en train".replace("&", "y")
        if c_ == "rama2":
            puntos.append(f'<rect x="{a - 6:.1f}" y="{b - 6:.1f}" width="12" height="12" fill="{color}" stroke="black" '
                          f'stroke-width="1.5"><title>{info}</title></rect>')
        else:
            puntos.append(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="5.5" fill="{color}" stroke="white" stroke-width="0.7">'
                          f'<title>{info}</title></circle>')
    leyenda = "".join(f'<li><span style="background:{COLOR_LETRA[l]}"></span>{n}</li>' for l, n in NOMBRE_LETRA.items())
    html = f"""<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ramas de actividad en el espacio de conceptos</title>
<style>body{{font-family:system-ui,sans-serif;margin:16px;color:#222}}h1{{font-size:1.2rem}}p{{max-width:70ch;color:#444}}
.fila{{display:flex;flex-wrap:wrap;gap:16px}}svg{{background:#fafafa;border:1px solid #ddd;max-width:100%;height:auto}}
ul{{list-style:none;padding:0;margin:0}}li{{margin:3px 0;font-size:.85rem}}li span{{display:inline-block;width:12px;height:12px;margin-right:6px;vertical-align:-1px}}
</style></head><body><h1>Ramas de actividad en el espacio de conceptos</h1>
<p>Proyección PaCMAP (2D) de los embeddings de las ramas del modelo s0 (150 épocas). Pasá el cursor sobre un punto para ver
la rama. El color es el sector; los cuadrados son ramas raras agregadas a su división de 2 dígitos. Las distancias son
aproximadas: la proyección es una visualización.</p>
<div class="fila"><svg viewBox="0 0 {w} {h}" width="{w}" height="{h}">{''.join(puntos)}</svg><ul>{leyenda}</ul></div></body></html>"""
    ruta.write_text(html)


# Al hacer clic en la leyenda se resalta el grupo (y se atenúa el resto) en vez de apagarlo; se pueden elegir varios.
# Los botones de arriba seleccionan una categoría entera. El marcador `{plot_id}` lo reemplaza plotly por el id del gráfico.
JS_RESALTAR = """
(function () {
  var gd = document.getElementById('{plot_id}');
  var sel = new Set(), ocultar = false;
  var n = gd.data.length, idx = [], base = [], grupos = {}, orden = [];
  for (var i = 0; i < n; i++) {
    idx.push(i);
    base.push(gd.data[i].meta.base);
    var g = gd.data[i].meta.grupo;
    if (!(g in grupos)) { grupos[g] = []; orden.push(g); }
    grupos[g].push(i);
  }
  var estado = document.createElement('span');
  estado.style.cssText = 'margin-left:8px;color:#555;font-size:12px';

  function aplicar() {
    var op = [], vis = [], sz = [], todo = sel.size === 0, nombres = [];
    for (var i = 0; i < n; i++) {
      var s = todo || sel.has(i);
      op.push(s ? 1 : 0.05);
      vis.push(s || !ocultar);
      sz.push(!todo && s ? base[i] * 1.6 : base[i]);
      if (!todo && s) nombres.push(gd.data[i].name);
    }
    Plotly.restyle(gd, {opacity: op, visible: vis, 'marker.size': sz}, idx);
    estado.textContent = todo ? 'Sin selección: se ven todos los grupos.' : 'Resaltado: ' + nombres.join('; ');
  }

  gd.on('plotly_legendclick', function (ev) {
    var i = ev.curveNumber;
    if (sel.has(i)) { sel.delete(i); } else { sel.add(i); }
    aplicar();
    return false;
  });
  gd.on('plotly_legenddoubleclick', function () { return false; });

  var panel = document.createElement('div');
  panel.style.cssText = 'font:13px system-ui,sans-serif;margin:8px 12px;line-height:2.1';
  function boton(texto, fn) {
    var b = document.createElement('button');
    b.textContent = texto;
    b.style.cssText = 'margin:0 4px 0 0;padding:2px 9px;border:1px solid #bbb;border-radius:4px;background:#f6f6f6;cursor:pointer';
    b.onclick = fn;
    panel.appendChild(b);
  }
  panel.appendChild(document.createTextNode('Resaltar categoría: '));
  var etiqueta = {'Provincia': 'Provincias', 'Ramas': 'Ramas (todos los sectores)'};
  orden.forEach(function (g) {
    var nombre = etiqueta[g] || g;
    boton(nombre, function () { sel = new Set(grupos[g]); aplicar(); });
  });
  boton('Quitar resaltado', function () { sel = new Set(); aplicar(); });
  var lbl = document.createElement('label');
  lbl.style.cssText = 'margin-left:10px;cursor:pointer';
  var cb = document.createElement('input');
  cb.type = 'checkbox';
  cb.onchange = function () { ocultar = cb.checked; aplicar(); };
  lbl.appendChild(cb);
  lbl.appendChild(document.createTextNode(' Ocultar lo no resaltado'));
  panel.appendChild(lbl);
  panel.appendChild(estado);
  var ayuda = document.createElement('div');
  ayuda.style.cssText = 'color:#555;font-size:12px;line-height:1.4';
  ayuda.textContent = 'Un clic en un elemento de la leyenda lo resalta y atenúa el resto; otro clic sobre el mismo lo quita. Se pueden resaltar varios a la vez.';
  panel.appendChild(ayuda);
  gd.parentNode.insertBefore(panel, gd);
  aplicar();
})();
"""


def mapa_3d(xyz, tabla, ruta, titulo: str = "Espacio de conceptos en 3D (PaCMAP, modelo s0)") -> None:
    """Visualización 3D interactiva (HTML con plotly): rotar, acercar, pasar el cursor por un token para ver su nombre y
    descripción, y apagar o prender cada grupo haciendo clic en la leyenda. Salarios y años de nacimiento van con un degradé
    por percentil o año; las ramas, coloreadas por sector; las provincias, por región."""
    import plotly.graph_objects as go

    cat = tabla["categoria"].to_numpy()
    token = tabla["token"].to_numpy()
    valor = tabla["valor"].to_numpy()
    letra = tabla["letra"].to_numpy()
    r32 = tabla["r32"].to_numpy()
    region = tabla["region"].to_numpy()
    desc = tabla["descripcion"].to_numpy()
    frec = tabla["frecuencia"].to_numpy()
    fig = go.Figure()

    def traza(mask, nombre, hover, grupo=None, **marker):
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            return
        fig.add_trace(go.Scatter3d(
            x=xyz[idx, 0], y=xyz[idx, 1], z=xyz[idx, 2], mode="markers", name=nombre, legendgroup=grupo or nombre,
            meta={"grupo": grupo or nombre, "base": float(marker["size"])},
            hovertext=[hover(i) for i in idx], hoverinfo="text", marker=dict(line=dict(width=0), **marker)))

    traza(cat == "salario", "Salario (degradé por percentil)", lambda i: f"{token[i]}<br>percentil {valor[i]}",
          size=4, color=valor[cat == "salario"].astype(float), colorscale="Viridis", opacity=0.95)
    traza(cat == "nacimiento", "Año de nacimiento (degradé por año)", lambda i: f"{token[i]}<br>nacidos en {valor[i]} (o agrupados)",
          size=4, color=valor[cat == "nacimiento"].astype(float), colorscale="Plasma", opacity=0.95)
    cmap = plt.get_cmap("tab10")
    for k, r in enumerate(sorted(set(region[cat == "provincia"]))):
        traza((cat == "provincia") & (region == r), f"Provincia: {r}", lambda i: f"{token[i]}<br>{desc[i]}<br>región {region[i]}",
              grupo="Provincia", size=6, color=matplotlib.colors.to_hex(cmap(k)), opacity=0.95)
    for l in sorted(NOMBRE_LETRA):
        traza((cat == "rama") & (letra == l), f"Rama: {NOMBRE_LETRA[l]}",
              lambda i: f"{token[i]}<br>{desc[i]}<br>sector: {NOMBRE_LETRA[int(letra[i])]}<br>división r32: {r32[i]}<br>{frec[i]:,} eventos en train",
              grupo="Ramas", size=3.5, color=COLOR_LETRA[l], opacity=0.9)
    traza(cat == "rama2", "Rama agregada a 2 dígitos (r32)",
          lambda i: f"{token[i]}<br>{desc[i]}<br>agrupa ramas finas poco frecuentes<br>división r32: {r32[i]}",
          size=6, color="black", symbol="diamond", opacity=0.95)
    posiciones = ["top center", "bottom center", "middle right", "middle left"]
    for c_, nombre, color in (("tamaño", "Tamaño de empresa", "#d62728"), ("antigüedad", "Antigüedad de la empresa", "#7f7f7f"),
                              ("sexo", "Sexo", "#1f77b4")):
        idx = np.flatnonzero(cat == c_)
        etiquetas = [f"{PREFIJO_CATEGORIA[c_]}: {desc[i]}" for i in idx]
        fig.add_trace(go.Scatter3d(x=xyz[idx, 0], y=xyz[idx, 1], z=xyz[idx, 2], mode="markers+text", name=nombre,
                                   meta={"grupo": nombre, "base": 7.0},
                                   text=etiquetas, textposition=[posiciones[k % 4] for k in range(len(idx))],
                                   textfont=dict(size=10),
                                   hovertext=[f"{token[i]}<br>{desc[i]}" for i in idx], hoverinfo="text",
                                   marker=dict(size=7, color=color, symbol="square")))
    eje = dict(showticklabels=False, showgrid=True, zeroline=False, title="", backgroundcolor="rgba(0,0,0,0)")
    fig.update_layout(title=titulo, template="plotly_white", height=760, legend=dict(itemsizing="constant", font=dict(size=11)),
                      scene=dict(xaxis=eje, yaxis=eje, zaxis=eje, aspectmode="data"), margin=dict(l=0, r=0, t=50, b=0))
    fig.write_html(ruta, include_plotlyjs="cdn", config={"displaylogo": False}, post_script=JS_RESALTAR)
