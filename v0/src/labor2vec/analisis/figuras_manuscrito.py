"""Figuras 2 a 7 y S1 del manuscrito (`informes/manuscrito_preliminar.md`), en PNG (300 dpi) y PDF.

    python -m labor2vec.analisis.figuras_manuscrito [--solo 2 3 S1 ...] [--salida informes/figuras]

Salen de lo que ya está calculado: los modelos pre-entrenados de `runs/*_vista/` (figura 2), los `resultados.json` de
las fases 6 y 6b (figura 3), `fase8/comparaciones_tiempo.json` (figura 4), las extracciones de `runs/fase7/` (figuras 5
y S1) y las tablas de `informes/fase9/` (figuras 6 y 7). La figura 2 tarda unos 2 minutos, la 5 otros 2 (DensMAP) y la
S1 unos 3 más la primera vez (UMAP y PaCMAP de las personas).

Estilo: rótulos en castellano con coma decimal; colores de la paleta categórica validada (azul, naranja, aqua: pasan los
controles de daltonismo con todos los pares), rampas de un solo tono para magnitudes y tinta neutra para lo demás.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402

from labor2vec.utiles.rutas import INFORMES, RAIZ  # noqa: E402

AZUL, NARANJA, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
TINTA, TINTA_2, TENUE = "#0b0b0b", "#52514e", "#898781"
GRILLA, EJE, CLARO = "#e1e0d9", "#c3c2b7", "#d9d8d2"
RAMPA_AZUL = LinearSegmentedColormap.from_list(
    "azul", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
RAMPA_NARANJA = LinearSegmentedColormap.from_list("naranja", ["#fbd9c8", "#f4a27f", "#eb6834", "#b8461b", "#7a2c0d"])
MM = 1 / 25.4
ANCHO = 180 * MM  # doble columna

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.labelsize": 7, "axes.labelcolor": TINTA_2, "axes.edgecolor": EJE,
    "axes.linewidth": 0.6, "xtick.color": TINTA_2, "ytick.color": TINTA_2, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.fontsize": 6.5, "legend.frameon": False, "savefig.dpi": 300,
    "pdf.fonttype": 42, "figure.facecolor": "white", "axes.facecolor": "white",
})
COMA = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))


def coma(v: float, dec: int = 2) -> str:
    return f"{v:.{dec}f}".replace(".", ",")


def guardar(fig, salida: Path, nombre: str) -> None:
    salida.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(salida / f"{nombre}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  {salida / nombre}.png y .pdf", flush=True)


def letra(ax, texto: str, x: float | None = None) -> None:
    """Antepone la letra del panel al título (llamar después de `set_title`); `x` se conserva por compatibilidad."""
    ax.set_title(f"{texto}   {ax.get_title('left')}", loc="left")


def grilla_x(ax) -> None:
    ax.grid(axis="x", color=GRILLA, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(COMA)


def tabla_md(ruta: Path, primera_columna: str) -> dict[str, dict[str, float]]:
    """Primera tabla markdown de `ruta` cuya primera columna se llama `primera_columna`: {fila: {columna: valor}}."""
    lineas = ruta.read_text().splitlines()
    for i, l in enumerate(lineas):
        celdas = [c.strip() for c in l.strip().strip("|").split("|")]
        if l.startswith("|") and celdas[0] == primera_columna:
            cols, out = celdas[1:], {}
            for f in lineas[i + 2:]:
                if not f.startswith("|"):
                    break
                c = [x.strip() for x in f.strip().strip("|").split("|")]
                out[c[0]] = {k: float(v.split()[0]) if v.split()[0] != "nan" else np.nan for k, v in zip(cols, c[1:])}
            return out
    raise KeyError(f"{ruta}: no hay tabla con la columna {primera_columna!r}")


# ---- figura 2: espacio de conceptos ---------------------------------------------------------------------------------
def figura_2(salida: Path) -> None:
    from labor2vec.analisis import figuras
    from labor2vec.analisis import metricas as m
    from labor2vec.analisis.conceptos import K_HUB, K_PROV, K_RAMA, MODELOS
    from labor2vec.analisis.embeddings import cargar_conceptos, matriz_embeddings
    from labor2vec.tokens.vocabulario import N_ESPECIALES

    c = cargar_conceptos()
    t = c.tabla
    emb = {n: matriz_embeddings(r)[N_ESPECIALES:] for n, r in MODELOS.items()}
    cat, valor = t["categoria"].to_numpy(), t["valor"].to_numpy()
    rng = np.random.default_rng(0)

    fig = plt.figure(figsize=(ANCHO, 180 * MM))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1], hspace=0.4, wspace=0.6)
    ga = gs[0, :2].subgridspec(1, 1)
    gb = gs[0, 2].subgridspec(3, 1, height_ratios=[1, 1, 0.06], hspace=0.62)

    # (a) mapa PaCMAP
    ax = fig.add_subplot(ga[0])
    xy = figuras.proyectar(emb["s0"])
    sel = cat == "rama"
    ax.scatter(*xy[sel].T, s=5, color=CLARO, linewidths=0, zorder=1)
    sel = cat == "rama2"
    ax.scatter(*xy[sel].T, s=14, marker="D", color=TINTA_2, edgecolors="white", linewidths=0.4, zorder=2)
    sel = cat == "salario"
    ax.scatter(*xy[sel].T, s=9, c=valor[sel], cmap=RAMPA_AZUL, vmin=-10, vmax=100, linewidths=0, zorder=3)
    sel = cat == "nacimiento"
    ax.scatter(*xy[sel].T, s=9, c=valor[sel], cmap=RAMPA_NARANJA, vmin=1915, vmax=2003, linewidths=0, zorder=3)
    sel = cat == "provincia"
    ax.scatter(*xy[sel].T, s=6, color=TINTA, linewidths=0, zorder=3)
    sel = np.isin(cat, ["tamaño", "antigüedad", "sexo"])
    ax.scatter(*xy[sel].T, s=16, marker="s", facecolors="white", edgecolors=TINTA, linewidths=0.7, zorder=4)

    def extremo(categoria, v, texto, dx, dy):
        i = np.flatnonzero((cat == categoria) & (valor == v))[0]
        ax.annotate(texto, xy[i], xytext=(dx, dy), textcoords="offset points", fontsize=6, color=TINTA_2,
                    ha="right" if dx < 0 else "left", arrowprops=dict(arrowstyle="-", color=TENUE, lw=0.5))

    extremo("salario", 1, "percentil 1", 8, -8)
    extremo("salario", 100, "percentil 100", -14, 2)
    extremo("nacimiento", 1930, "1930", 6, -6)
    extremo("nacimiento", 1990, "1990", 8, -6)

    def rotulo(mascara, texto, donde, dx=0):
        """Rótulo de un grupo junto a su caja: arriba, abajo, a la izquierda o a la derecha."""
        p = xy[mascara]
        (x0, y0), (x1, y1) = p.min(0), p.max(0)
        pos, off, ha, va = {"arriba": (((x0 + x1) / 2, y1), (0, 5), "center", "bottom"),
                            "abajo": (((x0 + x1) / 2, y0), (0, -5), "center", "top"),
                            "izquierda": ((x0, (y0 + y1) / 2), (-6, 0), "right", "center"),
                            "derecha": ((x1, (y0 + y1) / 2), (6, 0), "left", "center"),
                            "arriba_der": (((x0 + x1) / 2, y1), (0, 5), "left", "bottom")}[donde]
        ax.annotate(texto, pos, xytext=(off[0] + dx, off[1]), textcoords="offset points", fontsize=6.5, color=TINTA,
                    fontweight="bold", ha=ha, va=va)

    rotulo(cat == "salario", "Salario\n(101 percentiles)", "izquierda")
    rotulo(cat == "nacimiento", "Año de nacimiento (77)", "arriba_der")
    rotulo(np.isin(cat, ["rama", "rama2"]), "Ramas de actividad (296)", "arriba", dx=-5)
    rotulo(cat == "provincia", "Provincias (25)", "arriba")
    rotulo(np.isin(cat, ["tamaño", "antigüedad"]), "Tamaño y antigüedad\nde la empresa (8)", "abajo")
    rotulo(cat == "sexo", "Sexo (2)", "derecha")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    (x0, y0), (x1, y1) = xy.min(0), xy.max(0)
    ax.set_xlim(x0 - 0.04 * (x1 - x0), x1 + 0.38 * (x1 - x0))  # lugar a la derecha para el recuadro
    ax.set_ylim(y0 - 0.1 * (y1 - y0), y1 + 0.1 * (y1 - y0))
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color=CLARO, ms=4, label="rama (4 dígitos)"),
        Line2D([], [], marker="D", ls="", color=TINTA_2, ms=3.5, label="rama agregada (2 dígitos)")],
        loc="lower left", handletextpad=0.2, borderaxespad=0, fontsize=6)
    ax.set_title("Espacio de conceptos (PaCMAP, 150 épocas, semilla 0)")
    letra(ax, "a")

    # recuadro con las provincias, forma según la región (abajo a la derecha, donde no hay tokens)
    mp = cat == "provincia"
    ins = ax.inset_axes([0.77, 0.03, 0.23, 0.32])
    reg = t["region"].to_numpy()
    formas = {"GBA": "o", "Pampeana": "s", "NOA": "^", "NEA": "v", "Cuyo": "D", "Patagonia": "P"}
    for r_, f_ in formas.items():
        s_ = mp & (reg == r_)
        ins.scatter(*xy[s_].T, s=12, marker=f_, facecolors="white" if f_ in "^vD" else TINTA, edgecolors=TINTA,
                    linewidths=0.6, label=r_)
    ins.set_xticks([])
    ins.set_yticks([])
    for s in ins.spines.values():
        s.set_visible(True)
        s.set_color(EJE)
    ins.legend(fontsize=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handletextpad=0.1,
               columnspacing=0.6, borderaxespad=0.2, title="Provincias por región (ampliado)", title_fontsize=5.5)

    # (b) similitud entre percentiles y entre años
    for k, (categoria, titulo) in enumerate([("salario", "Salario (percentiles)"), ("nacimiento", "Año de nacimiento")]):
        axb = fig.add_subplot(gb[k])
        sel = np.flatnonzero((cat == categoria) & ((valor >= 1) if categoria == "salario" else True))
        sel = sel[np.argsort(valor[sel])]
        x = m.normalizar(emb["s0"][sel])
        im = axb.imshow(np.clip(x @ x.T, 0, 1), cmap=RAMPA_AZUL, vmin=0, vmax=1, interpolation="nearest")
        v = valor[sel]
        marcas = [1, 25, 50, 75, 100] if categoria == "salario" else [1930, 1950, 1970, 1990]
        pos = [int(np.argmin(np.abs(v - mk))) for mk in marcas]
        axb.set_xticks(pos, [str(mk) for mk in marcas])
        axb.set_yticks(pos, [str(mk) for mk in marcas])
        axb.tick_params(length=2, labelsize=5.5)
        axb.set_title(titulo, fontsize=6.5)
        for s in axb.spines.values():
            s.set_visible(False)
        if k == 0:
            letra(axb, "b")
    cax = fig.add_subplot(gb[2])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("similitud coseno entre tokens (negativos en 0)", fontsize=6)
    cb.ax.tick_params(labelsize=5.5, length=2)
    cb.ax.xaxis.set_major_formatter(COMA)
    cb.outline.set_visible(False)

    # (c) pureza observada contra el nulo
    axc = fig.add_subplot(gs[1, 0])
    letra_r = t["letra"].to_numpy()
    filas = [("Rama → sector", cat == "rama", letra_r, K_RAMA),
             ("Rama → división", cat == "rama", t["r32"].to_numpy(), K_RAMA),
             ("Provincia → región", cat == "provincia", reg, K_PROV)]
    for i, (nombre, mascara, etiqueta, k) in enumerate(filas):
        et = etiqueta[mascara]
        nb = m.vecinos(emb["s0"][mascara], k)
        nulo = np.array([np.mean((p := rng.permutation(et))[nb] == p[:, None]) for _ in range(1000)])
        lo, hi = np.percentile(nulo, [2.5, 97.5])
        axc.plot([lo, hi], [i, i], color=TENUE, lw=4, solid_capstyle="round", alpha=0.6, zorder=1)
        for n_ in ("s0", "s1", "s2"):
            nb = m.vecinos(emb[n_][mascara], k)
            axc.scatter(np.mean(et[nb] == et[:, None]), i - 0.12, s=18, color=AZUL, edgecolors="white",
                        linewidths=0.5, zorder=3)
        nb = m.vecinos(emb["ep30"][mascara], k)
        axc.scatter(np.mean(et[nb] == et[:, None]), i + 0.12, s=18, facecolors="white", edgecolors=AZUL,
                    linewidths=0.8, zorder=2)
    axc.set_yticks(range(len(filas)), [f[0] for f in filas])
    axc.set_ylim(len(filas) - 0.5, -0.5)
    axc.set_xlim(0, 0.6)
    axc.set_xlabel("pureza de las vecindades")
    grilla_x(axc)
    axc.legend(handles=[Line2D([], [], color=TENUE, lw=4, alpha=0.6, label="azar (95 %)"),
                        Line2D([], [], marker="o", ls="", color=AZUL, ms=4, label="150 épocas (3 semillas)"),
                        Line2D([], [], marker="o", ls="", mfc="white", mec=AZUL, ms=4, label="30 épocas")],
               loc="upper center", bbox_to_anchor=(0.2, -0.22), ncol=1, handlelength=1.2)
    axc.set_title("Pureza por sector y región")
    letra(axc, "c")

    # (d) reproducibilidad entre semillas por tipo de token
    axd = fig.add_subplot(gs[1, 1])
    grupos = [("Salario", cat == "salario"), ("Año de nac.", cat == "nacimiento"), ("Rama", cat == "rama"),
              ("Provincia", cat == "provincia"), ("Todos", np.ones(len(cat), bool))]
    for i, (nombre, mascara) in enumerate(grupos):
        d = {n_: m.distancias_coseno(emb[n_][mascara]) for n_ in ("s0", "s1", "s2")}
        rhos = [m.rho_modelos(d[a], d[b]) for a, b in (("s0", "s1"), ("s0", "s2"), ("s1", "s2"))]
        axd.plot([min(rhos), max(rhos)], [i, i], color=AZUL, lw=0.8, zorder=1)
        axd.scatter(rhos, [i] * 3, s=16, color=AZUL, edgecolors="white", linewidths=0.5, zorder=2)
    nulo = m.rho_modelos(m.distancias_coseno(emb["s0"]),
                         m.distancias_coseno(m.nulo_columnas_permutadas(emb["s0"], rng)))
    axd.axvline(nulo, color=TENUE, lw=0.8, ls="--")
    axd.text(nulo + 0.03, -0.3, "nulo", fontsize=6, color=TENUE)
    axd.set_yticks(range(len(grupos)), [g[0] for g in grupos])
    axd.set_ylim(len(grupos) - 0.5, -0.5)
    axd.set_xlim(-0.05, 1)
    axd.set_xlabel("Spearman entre distancias\n(3 pares de semillas)")
    grilla_x(axd)
    axd.set_title("Acuerdo entre semillas")
    letra(axd, "d")

    # (e) hubness: grado de entrada por grupo, 30 contra 150 épocas
    axe = fig.add_subplot(gs[1, 2])
    gr = {"r34": cat == "rama", "r32": cat == "rama2", "otros": ~np.isin(cat, ["rama", "rama2"])}
    nombres = {"r34": "ramas\n4 díg.", "r32": "ramas\nagregadas", "otros": "resto"}
    jit = np.random.default_rng(1)
    for j, (modelo, color, desp) in enumerate([("ep30", "#86b6ef", -0.18), ("s0", "#184f95", 0.18)]):
        g = m.grado_entrada(emb[modelo], K_HUB)
        for i, (clave, mascara) in enumerate(gr.items()):
            y = g[mascara]
            axe.scatter(i + desp + jit.uniform(-0.1, 0.1, len(y)), y + jit.uniform(-0.25, 0.25, len(y)), s=3,
                        color=color, linewidths=0, alpha=0.8)
            axe.plot([i + desp - 0.13, i + desp + 0.13], [np.median(y)] * 2, color=TINTA, lw=1.2)
    axe.axhline(K_HUB, color=TENUE, lw=0.6, ls="--")
    axe.set_xticks(range(3), [nombres[k] for k in gr])
    axe.yaxis.set_major_locator(MaxNLocator(integer=True))
    axe.set_ylabel(f"grado de entrada (k = {K_HUB})")
    axe.grid(axis="y", color=GRILLA, linewidth=0.5)
    axe.set_axisbelow(True)
    axe.legend(handles=[Line2D([], [], marker="o", ls="", color="#86b6ef", ms=4, label="30 épocas"),
                        Line2D([], [], marker="o", ls="", color="#184f95", ms=4, label="150 épocas"),
                        Line2D([], [], color=TINTA, lw=1.2, label="mediana")],
               loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3, handlelength=1, columnspacing=0.8)
    axe.set_title("Hubs y entrenamiento")
    letra(axe, "e")
    guardar(fig, salida, "fig2_conceptos")


# ---- figura 3: predicción -------------------------------------------------------------------------------------------
TAREAS = [("t1", "T1 · sale del registro", "AUL", INFORMES / "fase6/resultados.json", "aul"),
          ("t2", "T2 · quintil salarial", "QWK", INFORMES / "fase6b/t2/resultados.json", "qwk"),
          ("t3", "T3 · cambio de rama", "AUL", INFORMES / "fase6b/t3/resultados.json", "aul"),
          ("t4", "T4 · cambio de tamaño", "AUC macro", INFORMES / "fase6b/t4/resultados.json", "auc_macro")]
MODELOS_PRED = [("Transformer pre-entrenado (L2V)", "Transformer pre-entrenado"), ("GRU", "GRU"),
                ("Regresión logística", "Logística (proporciones)"), ("FFNN", "FFNN (proporciones)"),
                ("Logística con estado actual", "Logística con estado actual"), ("Persistencia", "Persistencia"),
                ("Transformer sin pre-entrenar", "Transformer sin pre-entrenar"),
                ("Tabla de vida (edad y sexo)", "Tabla de vida (edad y sexo)")]


def _modelos_de(ruta: Path) -> dict:
    d = json.loads(ruta.read_text())
    return d["y_2019"] if "y_2019" in d else d.get("y", d)


def figura_3(salida: Path) -> None:
    fig, ejes = plt.subplots(1, 4, figsize=(ANCHO, 62 * MM), sharey=True)
    for k, (ax, (tarea, titulo, metrica, ruta, clave)) in enumerate(zip(ejes, TAREAS)):
        d = _modelos_de(ruta)
        azar = d["Azar"][clave][0] if isinstance(d["Azar"][clave], list) else d["Azar"][clave]
        lim = {"t1": (0.49, 0.65), "t2": (0.70, 0.85), "t3": (0.49, 0.75), "t4": (0.49, 0.79)}[tarea]
        for i, (nombre, _) in enumerate(MODELOS_PRED):
            if nombre not in d:
                continue
            v = d[nombre][clave]
            v, lo, hi = (v if isinstance(v, list) else [v, v, v])
            if v < lim[0]:
                continue
            es_l2v = nombre.startswith("Transformer pre")
            color = AZUL if es_l2v else TINTA_2
            ax.plot([lo, hi], [i, i], color=color, lw=1.0)
            vacio = nombre == "Transformer sin pre-entrenar"
            ax.scatter(v, i, s=22 if es_l2v else 14, facecolors="white" if vacio else color, edgecolors=color,
                       linewidths=0.8, zorder=3)
            if es_l2v:
                ax.axvline(v, color=AZUL, lw=0.5, alpha=0.4, zorder=0)
            if vacio and tarea == "t2":
                ax.annotate("no convergió", (v, i), xytext=(0, -9), textcoords="offset points", fontsize=5.5,
                            color=TINTA_2, ha="center")
        ax.axvline(azar, color=TENUE, lw=0.6, ls="--")
        ax.set_title(titulo, fontsize=6.5)
        ax.set_xlabel(metrica)
        ax.set_xlim(*lim)
        if tarea == "t2":
            ax.text(0.705, 7.25, "tabla de vida: 0,14 · azar: −0,02", fontsize=5.5, color=TINTA_2)
        else:
            ax.text(azar + 0.003, 7.4, "azar", fontsize=5.5, color=TENUE)
        grilla_x(ax)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        letra(ax, "abcd"[k])
    ejes[0].set_yticks(range(len(MODELOS_PRED)), [m_[1] for m_ in MODELOS_PRED])
    ejes[0].invert_yaxis()
    fig.subplots_adjust(wspace=0.32)
    guardar(fig, salida, "fig3_prediccion")


# ---- figura 4: diferencias pareadas contra las líneas de base -------------------------------------------------------
def figura_4(salida: Path) -> None:
    d = json.loads((INFORMES / "fase8/comparaciones_tiempo.json").read_text())["con_tiempo"]
    bases = [("Regresión logística", "Logística"), ("FFNN", "FFNN"), ("Logística con estado actual", "Logística con estado"),
             ("Regresión logística + tiempo", "Logística + tiempo"), ("FFNN + tiempo", "FFNN + tiempo"),
             ("Logística con estado actual + tiempo", "Logística con estado + tiempo")]
    metricas = {"t1": "AUL", "t2": "QWK", "t3": "AUL", "t4": "AUC macro"}
    fig, ejes = plt.subplots(1, 4, figsize=(ANCHO, 60 * MM), sharey=True)
    for k, (ax, tarea) in enumerate(zip(ejes, metricas)):
        for j, (pref, color, forma, desp) in enumerate([(tarea, AZUL, "o", -0.14), (f"{tarea}_uv", NARANJA, "s", 0.14)]):
            for i, (b, _) in enumerate(bases):
                r = d.get(f"{pref}_{b}")
                if r is None:
                    continue
                y = i + (0.5 if i >= 3 else 0) + desp
                ax.plot([r["lo"], r["hi"]], [y, y], color=color, lw=1.0)
                ax.scatter(r["dif"], y, s=13, marker=forma, color=color, edgecolors="white", linewidths=0.4, zorder=3)
        ax.axvline(0, color=TINTA, lw=0.7)
        ax.axhspan(2.95, 6.05, color="#f4f3ef", zorder=0)
        ax.set_title(f"{TAREAS[k][1]}", fontsize=6.5)
        ax.set_xlabel(metricas[tarea])
        grilla_x(ax)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        letra(ax, "abcd"[k], 0.0)
    ejes[0].set_yticks([0, 1, 2, 3.5, 4.5, 5.5], [b[1] for b in bases])
    ejes[0].invert_yaxis()
    ejes[-1].legend(handles=[Line2D([], [], marker="o", color=AZUL, ms=4, label="con todos los atributos"),
                             Line2D([], [], marker="s", color=NARANJA, ms=4, label="sin atributos de último valor"),
                             Patch(color="#f4f3ef", label="líneas de base con variables de tiempo")],
                    loc="lower right", bbox_to_anchor=(1.0, 1.1), ncol=3, handlelength=1.5)
    fig.text(0.5, -0.1, "Diferencia transformer − línea de base: a la derecha del cero gana el transformer. "
             "Intervalos al 95 % por bootstrap pareado.", ha="center", fontsize=6, color=TINTA_2)
    fig.subplots_adjust(wspace=0.22)
    guardar(fig, salida, "fig4_controles")


# ---- figura 5: espacio de personas ----------------------------------------------------------------------------------
def figura_5(salida: Path) -> None:
    import polars as pl
    import umap

    from labor2vec.analisis.personas import CONCEPTOS, CORTE, N_FONDO, dias_de_mes

    p = dict(np.load(RAIZ / "runs/fase7/t1/personas.npz"))
    df = pl.read_parquet(CONCEPTOS)
    pct = dict(zip(df["id_trabajador"].to_list(), df["pct2015"].to_list()))
    salario = np.array([pct.get(int(i), np.nan) for i in p["test_id"]], dtype=float)
    print("  DensMAP de T1 (≈ 2 min)...", flush=True)
    cache = RAIZ / "runs/fase7/t1/densmap_test.npy"  # runs/ no se versiona
    if cache.exists():
        xy = np.load(cache)
    else:
        xy = umap.UMAP(n_neighbors=15, metric="cosine", densmap=True, random_state=0).fit_transform(
            p["test_vec"].astype("float32"))
        np.save(cache, xy)

    fig = plt.figure(figsize=(ANCHO, 118 * MM))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.15], hspace=0.6, wspace=0.45)
    # el DensMAP deja unos pocos puntos muy lejos: se muestra el 99 % central de cada coordenada
    lo, hi = np.percentile(xy, [0.5, 99.5], axis=0)
    y = p["test_y"].astype(float)
    mapas = [("a", "Salario medio", salario, RAMPA_AZUL, (0, 100), "percentil salarial en 2015"),
             ("b", "Tasa de salida (T1)", y, RAMPA_NARANJA, (0, 0.4), "proporción que sale en 2019")]
    for k, (l_, titulo, valores, rampa, (vmin, vmax), etiqueta) in enumerate(mapas):
        a = fig.add_subplot(gs[:, k])
        ok = ~np.isnan(valores)
        hb = a.hexbin(*xy[ok].T, C=valores[ok], reduce_C_function=np.mean, gridsize=40, mincnt=5, cmap=rampa,
                      vmin=vmin, vmax=vmax, extent=(lo[0], hi[0], lo[1], hi[1]), linewidths=0.2, edgecolors="white")
        a.set_xlim(lo[0], hi[0])
        a.set_ylim(lo[1], hi[1])
        cb = fig.colorbar(hb, ax=a, orientation="horizontal", shrink=0.85, pad=0.03, aspect=30)
        cb.set_label(etiqueta, fontsize=6)
        cb.ax.tick_params(labelsize=5.5, length=2)
        cb.ax.xaxis.set_major_formatter(COMA)
        cb.outline.set_visible(False)
        a.set_title(titulo)
        letra(a, l_)
        a.set_xticks([])
        a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)

    # (c) pureza de vecindades menos azar, desde la tabla de la fase 7
    axc = fig.add_subplot(gs[0, 2])
    filas = [ln for ln in (INFORMES / "fase7/resultados.md").read_text().splitlines() if ln.startswith("| ")]
    cols = [c.strip() for c in filas[0].strip("|").split("|")][1:]
    datos = {}
    for f in filas[1:4]:
        c = [x.strip() for x in f.strip("|").split("|")]
        datos[c[0]] = {k: tuple(float(z.strip("()")) for z in v.split()) for k, v in zip(cols, c[1:])}
    atributos = [("salario", "salario"), ("letra", "rama"), ("tamaño", "tamaño"), ("edad", "edad"),
                 ("antigüedad", "antigüedad"), ("sexo", "sexo"), ("y", "resultado")]
    series = [("T1 ajustado", "ajustado a T1", AZUL, "o"), ("T3 ajustado", "ajustado a T3", NARANJA, "s"),
              ("pre-entrenado sin ajustar", "pre-entrenado", AQUA, "^")]
    for j, (fila, nombre, color, forma) in enumerate(series):
        for i, (col, _) in enumerate(atributos):
            obs, azar = datos[fila][col]
            axc.scatter(obs - azar, i + (j - 1) * 0.22, s=13, marker=forma, color=color, edgecolors="white",
                        linewidths=0.4, label=nombre if i == 0 else None, zorder=3)
    axc.axvline(0, color=TINTA, lw=0.6)
    axc.set_yticks(range(len(atributos)), [a[1] for a in atributos])
    axc.invert_yaxis()
    axc.set_xlabel("pureza de 5 vecinos − azar")
    grilla_x(axc)
    axc.legend(loc="lower right", handletextpad=0.2)
    axc.set_title("Qué comparten los vecinos")
    letra(axc, "c", -0.3)

    # (d) saliency según la antigüedad del evento
    axd = fig.add_subplot(gs[1, 2])
    tramos = np.array([0, 3, 6, 12, 24, 60, 1e9])
    nombres_t = ["0-3", "3-6", "6-12", "12-24", "24-60", "60+"]
    for modelo, color, forma, nombre in [("t1", AZUL, "o", "T1"), ("t3", NARANJA, "s", "T3")]:
        s = dict(np.load(RAIZ / f"runs/fase7/{modelo}/saliency.npz"))
        persona = np.repeat(np.arange(len(s["largo"])), s["largo"])
        inicio = np.concatenate([[0], np.cumsum(s["largo"])])
        ev = np.arange(len(persona)) - inicio[persona] >= N_FONDO
        meses = (dias_de_mes(np.array([CORTE]))[0] - s["abspos"]) / 30.4
        b = np.digitize(meses, tramos[1:-1])
        razon = [(s["puntaje"][ev & (b == i)].sum() / s["puntaje"][ev].sum()) / ((ev & (b == i)).sum() / ev.sum())
                 for i in range(len(nombres_t))]
        axd.plot(range(len(nombres_t)), razon, color=color, lw=1.5, marker=forma, ms=4, label=nombre)
    axd.axhline(1, color=TENUE, lw=0.6, ls="--")
    axd.text(5, 1.04, "peso proporcional", fontsize=5.5, color=TENUE, ha="right")
    axd.set_xticks(range(len(nombres_t)), nombres_t)
    axd.set_xlabel("meses antes del corte")
    axd.set_ylabel("% de la prominencia / % de tokens")
    axd.yaxis.set_major_formatter(COMA)
    axd.grid(axis="y", color=GRILLA, linewidth=0.5)
    axd.legend(loc="upper right")
    axd.set_title("Peso de la historia según su antigüedad")
    letra(axd, "d", -0.3)
    fig.suptitle("Espacio de personas del modelo ajustado a T1 (DensMAP en celdas hexagonales, 18.337 personas de prueba)", x=0.06, y=1.0,
                 ha="left", fontsize=7.5, fontweight="bold", color=TINTA)
    guardar(fig, salida, "fig5_personas")


# ---- figuras 6 y 7: sondas y estabilidad ----------------------------------------------------------------------------
DINAMICA = [("antiguedad", "antigüedad en el empleo actual"), ("meses_desde_cambio_empresa", "meses desde cambio de empleador"),
            ("meses_desde_cambio_tamano", "meses desde cambio de tamaño"), ("meses_desde_cambio_rama", "meses desde cambio de rama"),
            ("cambios_empresa_12", "cambios de empleador, 12 m"), ("cambios_empresa_60", "cambios de empleador, 60 m"),
            ("cambios_tamano_60", "cambios de tamaño, 60 m"), ("cambios_rama_60", "cambios de rama, 60 m"),
            ("huecos_24", "meses sin registro, 24 m"), ("huecos_60", "meses sin registro, 60 m"),
            ("hay_dif_12", "¿salario registrado hace 12 m?"), ("hay_dif_36", "¿salario registrado hace 36 m?"),
            ("dif_salario_36", "variación salarial, 36 m")]


def _filas_dinamica(ax, variables):
    ax.set_yticks(range(len(variables)), [v[1] for v in variables])
    ax.invert_yaxis()
    ax.set_xlim(-0.02, 0.72)
    ax.set_xlabel("R² de la sonda lineal (prueba)")
    grilla_x(ax)


def figura_6(salida: Path) -> None:
    f9 = INFORMES / "fase9"
    conteos = tabla_md(f9 / "resultados_conteos.md", "espacio")
    extras = tabla_md(f9 / "resultados_extras.md", "encoder")
    controles = tabla_md(f9 / "resultados_controles.md", "entrada")
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(ANCHO, 82 * MM), gridspec_kw=dict(width_ratios=[1, 1], wspace=0.95))
    series = [("proporciones", "proporciones de tokens (sin tiempo)", "white", TENUE, "o"),
              ("log(1 + conteos) + tamaño", "log(1 + conteos) + largo de la historia", NARANJA, NARANJA, "s"),
              ("pre (vec)", "vector pre-entrenado", AZUL, AZUL, "o")]
    for i, (clave, _) in enumerate(DINAMICA):
        vals = [conteos[s[0]][f"dinámica: {clave}"] for s in series]
        axa.plot([min(vals), max(vals)], [i, i], color=GRILLA, lw=2, zorder=1)
        for (fila, nombre, fc, ec, forma), v in zip(series, vals):
            axa.scatter(v, i, s=18, marker=forma, facecolors=fc, edgecolors=ec, linewidths=0.8, zorder=3,
                        label=nombre if i == 0 else None)
    _filas_dinamica(axa, DINAMICA)
    axa.legend(loc="upper center", bbox_to_anchor=(0.3, -0.13), ncol=1, handletextpad=0.2)
    axa.set_title("El vector contra referencias sin información de tiempo")
    letra(axa, "a", -0.02)

    sub = [v for v in DINAMICA if f"dinámica: {v[0]}" in controles["pre (sin tocar)"]]
    series_b = [(controles, "pre (sin tocar)", "vector pre-entrenado", AZUL, AZUL, "o", 22, 0),
                (controles, "fechas permutadas", "con fechas permutadas", "white", AZUL, "^", 20, -0.22),
                (controles, "orden permutado", "con el orden permutado", "white", AZUL, "D", 14, 0.22),
                (extras, "pesos sin entrenar", "encoder sin entrenar", AQUA, AQUA, "X", 22, 0)]
    for i, (clave, _) in enumerate(sub):
        for tabla, fila, nombre, fc, ec, forma, tam, dy in series_b:
            axb.scatter(tabla[fila][f"dinámica: {clave}"], i + dy, s=tam, marker=forma, facecolors=fc, edgecolors=ec,
                        linewidths=0.8, zorder=3, label=nombre if i == 0 else None)
    _filas_dinamica(axb, sub)
    axb.legend(loc="upper center", bbox_to_anchor=(0.3, -0.13), ncol=1, handletextpad=0.2)
    axb.set_title("Controles: tiempo, orden y pre-entrenamiento")
    letra(axb, "b", -0.02)
    guardar(fig, salida, "fig6_sondas")


def figura_7(salida: Path) -> None:
    f9 = INFORMES / "fase9"
    texto = (f9 / "resultados_semillas.md").read_text()
    sondas = tabla_md(f9 / "resultados_semillas.md", "espacio")
    fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(ANCHO, 66 * MM),
                                        gridspec_kw=dict(width_ratios=[1.2, 0.9, 1.1], wspace=0.95))
    variables = [("antigüedad (años)", "antigüedad (años)"), ("dinámica: cambios_empresa_60", "cambios de empleador, 60 m"),
                 ("dinámica: meses_desde_cambio_empresa", "meses desde cambio de empleador"),
                 ("dinámica: cambios_rama_60", "cambios de rama, 60 m"), ("dinámica: huecos_60", "meses sin registro, 60 m"),
                 ("dinámica: hay_dif_36", "¿salario registrado hace 36 m?"), ("dinámica: cambios_tamano_60", "cambios de tamaño, 60 m")]
    for j, (sem, color, forma) in enumerate([("0", AZUL, "o"), ("1", NARANJA, "s"), ("2", AQUA, "^")]):
        for i, (col, _) in enumerate(variables):
            axa.scatter(sondas[f"pre semilla {sem}"][col], i + (j - 1) * 0.2, s=16, marker=forma, color=color, edgecolors="white",
                        linewidths=0.4, zorder=3, label=f"semilla {sem}" if i == 0 else None)
    axa.set_yticks(range(len(variables)), [v[1] for v in variables])
    axa.invert_yaxis()
    axa.set_xlim(0, 1)
    axa.set_xlabel("R² de la sonda (vector pre-entrenado)")
    grilla_x(axa)
    axa.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, handletextpad=0.1, columnspacing=0.8)
    axa.set_title("Lo que se lee del vector: estable")
    letra(axa, "a", -0.02)

    # (b) vecinos coincidentes entre semillas
    espacios = [("pre", "pre-entrenado"), ("t1", "ajustado a T1"), ("t3", "ajustado a T3")]
    for i, (esp, nombre) in enumerate(espacios):
        bloque = texto.split(f"### {esp}\n")[1].split("###")[0]
        filas = [ln for ln in bloque.splitlines() if ln.startswith("| ") and "y" in ln.split("|")[1]][1:]
        vec = [float(ln.split("|")[3].split()[0]) for ln in filas]
        axb.plot([min(vec), max(vec)], [i, i], color=AZUL, lw=0.8)
        axb.scatter(vec, [i] * len(vec), s=16, color=AZUL, edgecolors="white", linewidths=0.5, zorder=3)
    axb.axvline(0.0025, color=TENUE, lw=0.6, ls="--")
    axb.set_ylim(-0.6, 2.6)
    axb.text(0.012, 1.5, "azar: 0,25 %", fontsize=5.5, color=TENUE)
    axb.set_yticks(range(3), [e[1] for e in espacios])
    axb.invert_yaxis()
    axb.set_xlim(0, 0.6)
    axb.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:.0f} %"))
    axb.grid(axis="x", color=GRILLA, linewidth=0.5)
    axb.set_xlabel("de los 10 vecinos, coinciden")
    axb.set_title("Vecinos entre semillas")
    letra(axb, "b", -0.02)

    # (c) acuerdo de tipologías (ARI): muestreo contra semilla
    techo = [ln for ln in (f9 / "resultados_techo.md").read_text().splitlines() if ln.startswith("| ") and " a " in ln and "ARI" not in ln]
    nombres = ["tokens: mitades", "pre: mitades", "pre: semillas 0 y 1", "pre: semillas 0 y 2", "tokens contra pre"]
    colores = [TENUE, TENUE, AZUL, AZUL, TINTA_2]
    for i, (ln, nombre, color) in enumerate(zip(techo, nombres, colores)):
        c = [x.strip() for x in ln.strip("|").split("|")]
        media = float(c[1])
        lo, hi = (float(z) for z in c[2].split(" a "))
        axc.plot([lo, hi], [i, i], color=color, lw=1.0)
        axc.scatter(media, i, s=18, color=color, edgecolors="white", linewidths=0.5, zorder=3)
    axc.set_yticks(range(len(nombres)), nombres)
    axc.invert_yaxis()
    axc.set_xlim(0, 0.9)
    axc.set_xlabel("acuerdo de tipologías Ward, k = 8 (ARI)")
    grilla_x(axc)
    axc.legend(handles=[Line2D([], [], marker="o", color=TENUE, ms=4, label="cambia la muestra"),
                        Line2D([], [], marker="o", color=AZUL, ms=4, label="cambia la semilla"),
                        Line2D([], [], marker="o", color=TINTA_2, ms=4, label="cambia el espacio")],
               loc="upper center", bbox_to_anchor=(0.4, -0.2), ncol=2, handlelength=1.2, columnspacing=0.8)
    axc.set_title("Tipologías: inestables entre semillas")
    letra(axc, "c", -0.02)
    guardar(fig, salida, "fig7_estabilidad")


# ---- figura S1: tres proyecciones de cada espacio (como la Fig. A.5 de life2vec) ------------------------------------
def _proyecciones_personas(vec: np.ndarray) -> dict[str, np.ndarray]:
    """DensMAP, UMAP y PaCMAP de los vectores de prueba de T1, guardados en `runs/fase7/t1/` (≈ 3 min sin caché)."""
    import umap

    from labor2vec.analisis import figuras

    calculos = {"DensMAP": lambda: umap.UMAP(n_neighbors=15, metric="cosine", densmap=True, random_state=0).fit_transform(vec),
                "UMAP": lambda: umap.UMAP(n_neighbors=15, metric="cosine", random_state=0).fit_transform(vec),
                "PaCMAP": lambda: figuras.proyectar(vec)}
    out = {}
    for metodo, calcular in calculos.items():
        cache = RAIZ / f"runs/fase7/t1/{metodo.lower()}_test.npy"  # runs/ no se versiona
        if not cache.exists():
            print(f"  {metodo} de las personas de T1...", flush=True)
            np.save(cache, calcular())
        out[metodo] = np.load(cache)
    return out


def figura_s1(salida: Path) -> None:
    import umap

    from labor2vec.analisis import figuras
    from labor2vec.analisis.conceptos import MODELOS
    from labor2vec.analisis.embeddings import cargar_conceptos, matriz_embeddings
    from labor2vec.tokens.vocabulario import N_ESPECIALES

    t = cargar_conceptos().tabla
    cat, valor = t["categoria"].to_numpy(), t["valor"].to_numpy()
    emb = matriz_embeddings(MODELOS["s0"])[N_ESPECIALES:]
    conceptos = {"DensMAP": umap.UMAP(n_neighbors=15, metric="cosine", densmap=True, random_state=0).fit_transform(emb),
                 "UMAP": umap.UMAP(n_neighbors=15, metric="cosine", random_state=0).fit_transform(emb),
                 "PaCMAP": figuras.proyectar(emb)}
    p = dict(np.load(RAIZ / "runs/fase7/t1/personas.npz"))
    personas = _proyecciones_personas(p["test_vec"].astype("float32"))
    prob = 1 / (1 + np.exp(-p["test_logit"]))
    y = p["test_y"].astype(float)
    orden = np.random.default_rng(0).permutation(len(y))  # sin orden de dibujo que tape un color con otro

    fig = plt.figure(figsize=(ANCHO, 175 * MM))
    gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 0.42], hspace=0.28, wspace=0.12)
    filas = ["Conceptos\n(pre-entrenado, semilla 0)", "Personas de prueba:\npredicción de T1", "Personas de prueba:\nresultado real"]

    def limpio(ax, i, j, l_):
        ax.set_title(metodo)
        letra(ax, l_)
        if j == 0:
            ax.set_ylabel(filas[i], fontsize=6.5, color=TINTA, labelpad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(GRILLA)

    for j, metodo in enumerate(conceptos):
        # fila 1: conceptos, con la codificación de la figura 2a
        ax = fig.add_subplot(gs[0, j])
        xy = conceptos[metodo]
        for sel, kw in [(cat == "rama", dict(s=4, color=CLARO, zorder=1)),
                        (cat == "rama2", dict(s=10, marker="D", color=TINTA_2, edgecolors="white", linewidths=0.3, zorder=2)),
                        (cat == "salario", dict(s=6, c=valor[cat == "salario"], cmap=RAMPA_AZUL, vmin=-10, vmax=100, zorder=3)),
                        (cat == "nacimiento", dict(s=6, c=valor[cat == "nacimiento"], cmap=RAMPA_NARANJA, vmin=1915, vmax=2003, zorder=3)),
                        (cat == "provincia", dict(s=5, color=TINTA, zorder=3)),
                        (np.isin(cat, ["tamaño", "antigüedad", "sexo"]),
                         dict(s=12, marker="s", facecolors="white", edgecolors=TINTA, linewidths=0.6, zorder=4))]:
            kw.setdefault("linewidths", 0)
            ax.scatter(*xy[sel].T, **kw)
        limpio(ax, 0, j, "abc"[j])

        # filas 2 y 3: personas; se muestra el 99 % central de cada coordenada (DensMAP deja puntos muy lejos)
        xy = personas[metodo]
        lo, hi = np.percentile(xy, [0.5, 99.5], axis=0)
        ax = fig.add_subplot(gs[1, j])
        sc = ax.scatter(*xy[orden].T, c=prob[orden], s=0.4, cmap=RAMPA_AZUL, vmin=0.2, vmax=0.8, linewidths=0,
                        rasterized=True)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        limpio(ax, 1, j, "def"[j])
        ax = fig.add_subplot(gs[2, j])
        hb = ax.hexbin(*xy.T, C=y, reduce_C_function=np.mean, gridsize=30, mincnt=5, cmap=RAMPA_NARANJA, vmin=0,
                       vmax=0.4, extent=(lo[0], hi[0], lo[1], hi[1]), linewidths=0.2, edgecolors="white")
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        limpio(ax, 2, j, "ghi"[j])

    # columna de referencias: leyenda de conceptos y escalas de color
    ley = fig.add_subplot(gs[0, 3])
    ley.axis("off")
    ley.legend(handles=[
        Line2D([], [], marker="o", ls="", color="#3987e5", ms=4, label="salario (101 percentiles,\nclaro → bajo)"),
        Line2D([], [], marker="o", ls="", color="#eb6834", ms=4, label="año de nacimiento (77,\nclaro → antiguo)"),
        Line2D([], [], marker="o", ls="", color=CLARO, ms=4, label="rama, 4 dígitos (282)"),
        Line2D([], [], marker="D", ls="", color=TINTA_2, ms=3.5, label="rama agregada (14)"),
        Line2D([], [], marker="o", ls="", color=TINTA, ms=3, label="provincia (25)"),
        Line2D([], [], marker="s", ls="", mfc="white", mec=TINTA, ms=4, label="tamaño, antigüedad\ny sexo (10)")],
        loc="center left", borderaxespad=0, fontsize=5.8, handletextpad=0.3, labelspacing=0.9)
    for fila, mapeable, etiqueta in [(1, sc, "probabilidad estimada\nde salir en 2019\n(sin calibrar)"),
                                     (2, hb, "proporción que sale\nen 2019 (celdas\ncon 5 o más personas)")]:
        cax = fig.add_subplot(gs[fila, 3].subgridspec(1, 3, width_ratios=[0.1, 0.12, 0.78])[1])
        cb = fig.colorbar(mapeable, cax=cax)
        cb.set_label(etiqueta, fontsize=5.8, labelpad=4)
        cb.ax.tick_params(labelsize=5.5, length=2)
        cb.ax.yaxis.set_major_formatter(COMA)
        cb.outline.set_visible(False)
    guardar(fig, salida, "figS1_proyecciones")


FIGURAS = {"2": figura_2, "3": figura_3, "4": figura_4, "5": figura_5, "6": figura_6, "7": figura_7, "S1": figura_s1}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--solo", nargs="*", default=list(FIGURAS), help="figuras a generar (2 a 7 y S1)")
    ap.add_argument("--salida", type=Path, default=INFORMES / "figuras")
    args = ap.parse_args(argv)
    for n in args.solo:
        print(f"figura {n}", flush=True)
        FIGURAS[n](args.salida)


if __name__ == "__main__":
    main()
