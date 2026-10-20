"""Esquema de labor2vec: de la historia laboral de un trabajador (MLER) al encoder y su decodificador.

Es el análogo, con nuestros datos, del esquema de la figura 1 de life2vec (Savcisens et al., 2023): una línea de tiempo
de eventos con su información posicional (edad, posición absoluta, segmento), las tablas de origen de dos eventos, y a la
derecha el recorrido de la secuencia por el encoder hasta la representación comprimida y el decodificador.

El trabajador es un ejemplo ilustrativo (valores inventados); las posiciones (edad, días desde 1996-01-01 y segmento)
se calculan con las mismas funciones que usa el modelo (`tokens.documento`).

    python -m labor2vec.analisis.esquema [--salida informes/figuras]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle  # noqa: E402

from labor2vec.tokens.documento import CICLO_SEGMENTOS, dias_de_mes, edad_en_anios  # noqa: E402
from labor2vec.utiles.rutas import INFORMES  # noqa: E402

AZUL, AZUL_CLARO, AZUL_TABLA = "#2f5aa8", "#a9c3ea", "#dce7f8"
NARANJA, NARANJA_CLARO = "#e08a1e", "#fbe8cc"
GRIS, GRIS_OSC = "#8a94a6", "#4b5568"
FNACIM = 1980
W, H = 330, 110
MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]


def mes_idx(anio: int, mes: int) -> int:
    return (anio - 1996) * 12 + mes - 1


def etiqueta(m: int) -> str:
    return f"{MESES[m % 12]} {1996 + m // 12}"


def x_de(m: float) -> float:
    return 24 + 0.76 * m


# trabajador de ejemplo: relación A (2003-03 a 2008-08), brecha, relación B (2011-03 a 2015-12) y una relación C en paralelo
REL_A = range(mes_idx(2003, 3), mes_idx(2008, 8) + 1)
REL_B = range(mes_idx(2011, 3), mes_idx(2015, 12) + 1)
REL_C = range(mes_idx(2014, 2), mes_idx(2014, 8) + 1)
SELECCION = [mes_idx(2003, 3), mes_idx(2005, 6), mes_idx(2008, 8), mes_idx(2011, 3), mes_idx(2013, 3), mes_idx(2014, 5),
             mes_idx(2015, 12)]
Y_LINEA, Y_EDAD, Y_POS, Y_SEG = 47, 31, 22, 13


def eventos() -> list[tuple[int, str]]:
    """Eventos del ejemplo ordenados por (mes, relación): [(mes_idx, relación)]."""
    ev = [(m, "A") for m in REL_A] + [(m, "B") for m in REL_B] + [(m, "C") for m in REL_C]
    return sorted(ev, key=lambda e: (e[0], e[1]))


def caja(ax, x, y, w, h, fc, ec="none", texto=None, color="white", tam=8, negrita=True, lw=1.0, ls="-", z=2, r=1.0,
         alpha=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls,
                                zorder=z, alpha=alpha))
    if texto:
        ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center", fontsize=tam, color=color, zorder=z + 1,
                fontweight="bold" if negrita else "normal", linespacing=1.25)


def punteada(ax, xs, ys, color=GRIS, lw=1.2):
    ax.plot(xs, ys, color=color, lw=lw, ls=(0, (1, 1.6)), zorder=1)


def tabla(ax, x, y, titulo, filas, fila_sel, xc_chip):
    """Tabla de origen de un evento: cabecera, tres filas (la del medio es la que se tokeniza) y la fecha de cada una."""
    cols = ["Salario", "Tamaño", "Rama", "Provincia", "Antig."]
    anchos = [10, 15, 27, 17, 14]
    w, alto_f = sum(anchos), 5.6
    ax.text(x + w / 2, y + 4 * alto_f + 3.2, titulo, ha="center", fontsize=8.5, fontweight="bold", color=AZUL)
    caja(ax, x, y + 3 * alto_f, w, alto_f, AZUL, texto=None, r=0.8)
    xx = x
    for c, a in zip(cols, anchos):
        ax.text(xx + a / 2, y + 3.5 * alto_f, c, color="white", fontsize=6.8, ha="center", va="center", fontweight="bold")
        xx += a
    for i, (fecha, vals) in enumerate(filas):
        yy = y + (2 - i) * alto_f
        sel = i == fila_sel
        ax.add_patch(Rectangle((x, yy), w, alto_f, fc=AZUL_TABLA if sel else "#f3f6fc", ec="none", zorder=2))
        xx = x
        for v, a in zip(vals, anchos):
            ax.text(xx + a / 2, yy + alto_f / 2, v, ha="center", va="center", fontsize=6.6, zorder=3,
                    color="black" if sel else "#9aa5b8", fontweight="bold" if sel else "normal")
            xx += a
        ax.text(x - 1.5, yy + alto_f / 2, fecha, ha="right", va="center", fontsize=7,
                color=AZUL if sel else "#9aa5b8", fontweight="bold" if sel else "normal")
    ax.add_patch(Rectangle((x, y), w, 4 * alto_f - alto_f, fc="none", ec=AZUL, lw=1.1, zorder=4))
    # conector punteado hacia el evento
    yb = y
    punteada(ax, [x + w / 2, x + w / 2, xc_chip, xc_chip], [yb, yb - 3, yb - 3, Y_LINEA + 3.2], AZUL, 1.3)


def lado_derecho(ax):
    """Recorrido de la secuencia por el modelo (equivalente a la columna derecha del esquema original)."""
    x0, ancho = 280, 48
    franjas = [("Edad", 100), ("Posición absoluta", 93), ("Segmento", 86), ("Token del evento", 79)]
    for nombre, y in franjas:
        ax.add_patch(Rectangle((x0, y), ancho, 6, fc="#eef1f6", ec=GRIS, lw=0.8, zorder=2, hatch="||||"))
        caja(ax, x0 + 9, y + 1, ancho - 18, 4, "white", texto=nombre, color="black", tam=8, negrita=False, r=0.4,
             z=3)
    # bandas de color en el token
    for i, c in enumerate(["#9ec5e8", "#b9d6a8", "#e8c98e", "#c9a8d8", "#e8a8a8", "#cfd3da"]):
        ax.add_patch(Rectangle((x0 + 1 + i * 2.8, 79.2), 2.0, 5.6, fc=c, ec="none", zorder=2))
    ax.add_patch(Rectangle((x0, 79), ancho, 6, fc="none", ec=GRIS, lw=0.8, zorder=4))
    caja(ax, x0 + 8, 80.3, ancho - 16, 3.4, "white", texto="Token del evento", color="black", tam=8, negrita=False,
         r=0.4, z=5)

    def abanico(y_arriba, y_abajo, x_a=(x0, x0 + ancho), x_b=(x0 + 6, x0 + ancho - 6)):
        ax.add_patch(Polygon([(x_a[0], y_arriba), (x_a[1], y_arriba), (x_b[1], y_abajo), (x_b[0], y_abajo)],
                             fc="#c9d1de", ec="none", alpha=0.8, zorder=1))

    abanico(79, 66)
    caja(ax, x0 + 4, 58, ancho - 8, 8, "#5f7396", texto="ENCODER", tam=11)
    abanico(58, 52, (x0 + 4, x0 + ancho - 4), (x0 + 4, x0 + ancho - 4))
    for i in range(5):
        ax.add_patch(Rectangle((x0 + i * ancho / 5, 44), ancho / 5, 8, fc="#eef1f6", ec=GRIS, lw=0.6, hatch="||||",
                               zorder=2))
    ax.text(x0 + ancho / 2, 48, "…", ha="center", va="center", fontsize=9, zorder=5)
    abanico(44, 36, (x0, x0 + ancho), (x0 + 4, x0 + ancho - 4))
    caja(ax, x0 + 4, 28, ancho - 8, 8, "#5f7396", texto="ENCODER", tam=11)
    ax.add_patch(Polygon([(x0 + 4, 28), (x0 + ancho - 4, 28), (x0 + ancho / 2 + 1.2, 24), (x0 + ancho / 2 - 1.2, 24)],
                         fc="#c9d1de", ec="none", alpha=0.8, zorder=1))
    caja(ax, x0 + ancho / 2 - 2.2, 17.5, 4.4, 8, NARANJA, r=1.6)
    caja(ax, x0 + 4, 3, ancho - 8, 10, "#5f7396", texto="DECODER", tam=11)
    ax.plot([x0 + ancho / 2] * 2, [17.5, 13], color=GRIS_OSC, lw=1.0, zorder=1)
    # rótulos de la izquierda de la columna
    for y, txt in ((79, "Secuencia de vida\ncruda\n(con posiciones)"), (41, "Secuencia\ncontextualizada"),
                   (16.5, "Secuencia comprimida\n(representación)")):
        caja(ax, 239, y - 1, 38, 11 if y != 41 else 8, "white", ec=GRIS, texto=txt, color="black", tam=7.2,
             negrita=False, lw=0.8, r=0.8)
    ax.text(x0 + ancho / 2, 108.5, "Modelo (backbone de life2vec)", ha="center", fontsize=8.5, fontweight="bold",
            color=GRIS_OSC)
    ax.text(x0 + ancho / 2, -2.6,
            "pre-entrenamiento: MLM + SOP\najuste fino T1: P(sin registro en 2019)", ha="center", va="top", fontsize=7.3,
            color=GRIS_OSC)


def dibujar() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(22, 7.4))
    ax.set_xlim(0, W)
    ax.set_ylim(-9, H)
    ax.axis("off")
    ev = eventos()

    # --- línea de tiempo: un evento por (relación, mes) ---
    ax.plot([x_de(0) - 3, x_de(239) + 4], [Y_LINEA, Y_LINEA], color=GRIS, lw=1.2, zorder=1)
    ax.text(x_de(0) - 3, Y_LINEA + 6, "1 ENE 1996", fontsize=8.5, color=AZUL, fontweight="bold", ha="left")
    ax.text(x_de(239) + 0.5, Y_LINEA + 6, "31 DIC 2015", fontsize=8.5, color=AZUL, fontweight="bold", ha="right")
    ax.text(x_de(239) + 0.5, Y_LINEA + 9.2, "(corte: mes 239)", fontsize=7.2, color=GRIS_OSC, ha="right")
    sel = set(SELECCION)
    for m, rel in ev:
        if m in sel and rel != "C" or (m - (REL_A.start if rel == "A" else REL_B.start if rel == "B" else REL_C.start)) % 3 == 0:
            desplazar = 1.0 if rel == "C" else 0.0
            ax.add_patch(FancyBboxPatch((x_de(m) - 0.7 + desplazar, Y_LINEA - 2.6), 1.5, 5.2,
                                        boxstyle="round,pad=0,rounding_size=0.5", fc=AZUL_CLARO if m not in sel else
                                        "#7fa6dc", ec=AZUL, lw=0.7, zorder=3))
    # brecha sin eventos
    xa, xb = x_de(REL_A.stop - 1) + 2, x_de(REL_B.start) - 2
    ax.annotate("", xy=(xa, Y_LINEA - 6.5), xytext=(xb, Y_LINEA - 6.5),
                arrowprops=dict(arrowstyle="<->", color=GRIS, lw=0.9))
    ax.text((xa + xb) / 2, Y_LINEA - 9.5, "sin registro: no hay eventos", ha="center", fontsize=7.2, color=GRIS_OSC,
            style="italic")
    ax.text(x_de(REL_C.start + 3) + 1, Y_LINEA - 4.6, "dos relaciones\nel mismo mes", ha="center", fontsize=6.8,
            color=GRIS_OSC, style="italic", va="top")

    # --- fondo (background) ---
    caja(ax, 0.5, Y_LINEA - 6.3, 20, 12.6, "#eef1f6", ec=GRIS, lw=0.8, r=1.0)
    ax.text(10.5, Y_LINEA + 3.2, "[CLS]", ha="center", fontsize=6.8, color=GRIS_OSC)
    ax.text(10.5, Y_LINEA + 0.2, "SEXO_V", ha="center", fontsize=6.8, color=GRIS_OSC)
    ax.text(10.5, Y_LINEA - 2.8, "NAC_1980", ha="center", fontsize=6.8, color=GRIS_OSC)
    ax.text(10.5, Y_LINEA - 5.6, "[SEP]", ha="center", fontsize=6.8, color=GRIS_OSC)
    ax.text(10.5, Y_LINEA + 8.2, "fondo", ha="center", fontsize=7.2, color=GRIS_OSC, fontweight="bold")

    # --- ventana de predicción (a futuro) ---
    caja(ax, 210, Y_LINEA - 11, 26, 26, NARANJA_CLARO, ec=NARANJA, lw=1.2, ls=(0, (3, 2)), r=1.2, z=1)
    ax.text(223, Y_LINEA + 11.3, "2016 – 2019", ha="center", fontsize=8.2, fontweight="bold", color=NARANJA)
    ax.text(223, Y_LINEA + 3.6, "¿algún mes registrado\nen 2019?", ha="center", fontsize=7.2, color="#7a4a0a",
            linespacing=1.3)
    ax.text(223, Y_LINEA - 5, "y = 1 si no", ha="center", fontsize=7.8, fontweight="bold", color=NARANJA)
    ax.text(223, Y_LINEA - 8.7, "(no entra al modelo)", ha="center", fontsize=6.8, color="#9b6a2a", style="italic")
    punteada(ax, [x_de(239) + 3.5] * 2, [Y_LINEA - 13, Y_LINEA + 14], NARANJA, 1.3)

    # --- filas posicionales ---
    for y, txt in ((Y_EDAD, "Edad (años)"), (Y_POS, "Posición absoluta (días)"), (Y_SEG, "Segmento")):
        ax.plot([x_de(0) - 3, x_de(239) + 4], [y, y], color="#c6ccd8", lw=1.0, zorder=1)
        ax.text(x_de(0) - 4.5, y + 3.2, txt, ha="left", fontsize=8.2, color=GRIS_OSC)
    seg_de = {e: CICLO_SEGMENTOS[i % len(CICLO_SEGMENTOS)] for i, e in enumerate(ev)}
    for m in SELECCION:
        rel = next(r for mm, r in ev if mm == m)
        x = x_de(m) + 0.05
        edad = int(edad_en_anios(np.array([m]), FNACIM)[0])
        dias = int(dias_de_mes(np.array([m]))[0])
        punteada(ax, [x] * 2, [Y_LINEA - 2.8, Y_SEG - 3.2])
        for y, txt in ((Y_EDAD, str(edad)), (Y_POS, f"{dias:,}".replace(",", ".")), (Y_SEG, str(seg_de[(m, rel)]))):
            caja(ax, x - 4.6, y - 2.9, 9.2, 5.8, "white", ec=GRIS, texto=txt, color="black", tam=8.2, negrita=False,
                 lw=0.9, r=0.8, z=4)

    # --- tablas de origen de dos eventos ---
    m1, m2 = mes_idx(2005, 6), mes_idx(2014, 5)
    tabla(ax, 47, 70, "REGISTRO LABORAL (MLER): relación A", [
        ("may 2005", ["p56", "10-50", "Comercio minorista", "Córdoba", "2001-05"]),
        ("jun 2005", ["p57", "10-50", "Comercio minorista", "Córdoba", "2001-05"]),
        ("jul 2005", ["p57", "10-50", "Comercio minorista", "Córdoba", "2001-05"])], 1, x_de(m1))
    tabla(ax, 143, 70, "REGISTRO LABORAL (MLER): relación B", [
        ("abr 2014", ["p81", "50-200", "Transporte automotor", "Córdoba", "2006-10"]),
        ("may 2014", ["p82", "50-200", "Transporte automotor", "Córdoba", "2006-10"]),
        ("jun 2014", ["p82", "50-200", "Transporte automotor", "Córdoba", "2006-10"])], 1, x_de(m2))

    # nota de tokenización
    ax.text(48, 105.8, "Cada fila (relación, mes) es un evento de 6 tokens, p. ej.:", fontsize=7.8, color=GRIS_OSC)
    ax.text(48 + 66, 105.8, "REM_057  TAM_3  RAMA_5211  PROV_14  ANT_2  [SEP]", fontsize=8.2, color=AZUL,
            fontweight="bold", family="monospace")
    ax.text(0.8, -7.2, "Ejemplo ilustrativo (valores inventados) de un trabajador nacido en 1980 con historia 1996-2015. "
            "Edad, días desde 1996-01-01 y segmento (ciclo 2, 3, 1) se calculan como en el modelo.", fontsize=7.2,
            color=GRIS_OSC, style="italic")
    lado_derecho(ax)
    return fig


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--salida", default=str(INFORMES / "figuras"))
    args = ap.parse_args(argv)
    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    fig = dibujar()
    for ext in ("png", "svg"):
        fig.savefig(salida / f"esquema_labor2vec.{ext}", dpi=170 if ext == "png" else None, bbox_inches="tight",
                    facecolor="white")
    print(f"escrito en {salida}")


if __name__ == "__main__":
    main()
