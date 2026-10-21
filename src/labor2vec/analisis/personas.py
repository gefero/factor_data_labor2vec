"""Fase 7, paso local: conceptos de las personas, mapas, TCAV y saliency (plans/PLAN_fase_7.md).

    python -m labor2vec.analisis.personas conceptos                      # tabla de conceptos de la cohorte de T1
    python -m labor2vec.analisis.personas analizar [--entrada runs/fase7]

`analizar` lee lo que dejó `analisis.extraer` en `runs/fase7/{t1,t3,pre}/` (personas.npz y saliency.npz) y escribe las
tablas en `informes/fase7/resultados.md` y las figuras en `informes/fase7/`.
"""

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from labor2vec.analisis import figuras, tcav  # noqa: E402
from labor2vec.analisis import metricas as m  # noqa: E402
from labor2vec.analisis.embeddings import cargar_conceptos  # noqa: E402
from labor2vec.tareas.t1 import ANIO_CORTE, CORTE, _ventana  # noqa: E402
from labor2vec.tokens.documento import N_FONDO, dias_de_mes  # noqa: E402
from labor2vec.utiles.duck import conectar  # noqa: E402
from labor2vec.utiles.rutas import INFORMES, PANEL, RAIZ, T1, TAREAS  # noqa: E402

CONCEPTOS = TAREAS / "conceptos_personas.parquet"
ENTRADA = RAIZ / "runs/fase7"
SALIDA = INFORMES / "fase7"
MODELOS = {"t1": "T1 ajustado", "t3": "T3 ajustado", "pre": "pre-entrenado sin ajustar"}
N_LETRAS = 6


# ---- conceptos ------------------------------------------------------------------------------------------------------
def sql_conceptos(panel: Path, t1: Path) -> str:
    """Una fila por persona de la cohorte, con su estado en 2015 (relación principal: la de más meses del año) y el primer
    mes que aparece en el panel (para la antigüedad en el empleo registrado)."""
    a0, a1 = _ventana(ANIO_CORTE)
    return f"""
WITH coh AS (SELECT id_trabajador, split, sexo, edad_corte FROM '{t1}'),
filas AS (
    SELECT p.id_trabajador, p.id_relacion, p.letra, p.tramo_empleo, p.rem_tot, p.pct_rem_mes, p.mes_idx
    FROM '{panel}' p JOIN coh USING (id_trabajador) WHERE p.mes_idx <= {CORTE}),
rel AS (
    SELECT id_trabajador, id_relacion, max(letra) AS letra, max(tramo_empleo) AS tramo, count(*) AS meses, sum(rem_tot) AS rem
    FROM filas WHERE mes_idx BETWEEN {a0} AND {a1} GROUP BY ALL),
principal AS (
    SELECT id_trabajador, letra, tramo FROM (
        SELECT *, row_number() OVER (PARTITION BY id_trabajador ORDER BY meses DESC, rem DESC, id_relacion) AS rk FROM rel)
    WHERE rk = 1),
hist AS (
    SELECT id_trabajador, min(mes_idx) AS primer_mes,
           avg(pct_rem_mes) FILTER (WHERE mes_idx BETWEEN {a0} AND {a1} AND pct_rem_mes > 0) AS pct2015
    FROM filas GROUP BY id_trabajador)
SELECT c.id_trabajador, c.split, c.sexo, c.edad_corte, a.letra, a.tramo, h.pct2015,
       ({CORTE} - h.primer_mes) / 12.0 AS antig_anios
FROM coh c LEFT JOIN principal a USING (id_trabajador) LEFT JOIN hist h USING (id_trabajador)
ORDER BY c.id_trabajador
"""


def construir_conceptos(panel: Path = PANEL, t1: Path = T1, salida: Path = CONCEPTOS) -> pl.DataFrame:
    df = conectar().execute(sql_conceptos(panel, t1)).pl()
    Path(salida).parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(salida)
    return df


def _terciles(x: np.ndarray, train: np.ndarray) -> np.ndarray:
    """0 (bajo), 1, 2 (alto) con los cortes de train; -1 si falta el dato."""
    ok = ~np.isnan(x)
    cortes = np.quantile(x[train & ok], [1 / 3, 2 / 3])
    return np.where(ok, np.searchsorted(cortes, np.nan_to_num(x), side="right"), -1)


def conceptos_de(df: pl.DataFrame) -> tuple[dict, dict]:
    """(etiquetas, grupos): `etiquetas` = {nombre: códigos enteros por persona} para medir pureza de vecindades, y
    `grupos` = {nombre: máscara booleana} con los conceptos de TCAV (cada uno contra el resto de la cohorte)."""
    train = (df["split"] == "train").to_numpy()
    sexo, letra = df["sexo"].to_numpy(), df["letra"].fill_null(-1).to_numpy()
    tramo = df["tramo"].fill_null(0).to_numpy()
    t_sal = _terciles(df["pct2015"].to_numpy().astype(float), train)
    t_ant = _terciles(df["antig_anios"].to_numpy().astype(float), train)
    t_edad = _terciles(df["edad_corte"].to_numpy().astype(float), train)
    valores, cuentas = np.unique(letra[letra >= 0], return_counts=True)
    top = valores[np.argsort(-cuentas)][:N_LETRAS]
    chica = tramo[tramo > 0].max()
    etiquetas = {"sexo": sexo, "letra": np.where(np.isin(letra, top), letra, -1),
                 "tamaño": np.where(tramo == 1, 2, np.where(tramo == chica, 0, np.where(tramo > 0, 1, -1))),
                 "salario": t_sal, "antigüedad": t_ant, "edad": t_edad}
    grupos = {"mujer": sexo == 1, "empresa grande (>200)": tramo == 1, "empresa chica (≤10)": tramo == chica,
              "salario alto": t_sal == 2, "salario bajo": t_sal == 0, "antigüedad alta": t_ant == 2,
              "antigüedad baja": t_ant == 0, "mayor": t_edad == 2, "joven": t_edad == 0}
    grupos |= {f"rama letra {int(a)}": letra == a for a in top}
    return etiquetas, grupos


# ---- lectura --------------------------------------------------------------------------------------------------------
def cargar(entrada: Path) -> dict:
    out = {}
    for nombre in MODELOS:
        ruta = Path(entrada) / nombre / "personas.npz"
        if ruta.exists():
            out[nombre] = {"personas": dict(np.load(ruta)),
                           "saliency": dict(np.load(ruta.with_name("saliency.npz"))) if ruta.with_name("saliency.npz").exists() else None}
    return out


def alinear(ids: np.ndarray, df: pl.DataFrame, columnas: dict) -> dict:
    """Reordena arreglos indexados por la cohorte completa (`df`) al orden de `ids`."""
    pos = {int(i): k for k, i in enumerate(df["id_trabajador"].to_numpy())}
    k = np.array([pos[int(i)] for i in ids])
    return {n: v[k] for n, v in columnas.items()}


# ---- mapas ----------------------------------------------------------------------------------------------------------
def pureza_vecindades(vec: np.ndarray, y: np.ndarray, etiquetas: dict, k: int = 5, n: int = 4000, n_perm: int = 100) -> dict:
    """Pureza de los k vecinos (coseno) para `y` y cada concepto, sobre una submuestra de `n` personas."""
    rng = np.random.default_rng(0)
    s = np.sort(rng.choice(len(vec), min(n, len(vec)), replace=False))
    out = {"y": m.pureza(vec[s], y[s], k, n_perm, rng)}
    for nombre, e in etiquetas.items():
        ok = e[s] >= 0
        out[nombre] = m.pureza(vec[s][ok], e[s][ok], k, n_perm, rng)
    return out


def proyecciones(vec: np.ndarray) -> dict:
    import umap
    xy_dens = umap.UMAP(n_neighbors=15, metric="cosine", densmap=True, random_state=0).fit_transform(vec.astype("float32"))
    return {"DensMAP": xy_dens, "PaCMAP": figuras.proyectar(vec)}


def figura_mapa(xy: np.ndarray, colores: dict, ruta: Path, titulo: str) -> None:
    """Un panel por variable: continuas con escala de color, categóricas con una paleta discreta."""
    n = len(colores)
    fig, ejes = plt.subplots(2, -(-n // 2), figsize=(4.2 * -(-n // 2), 8))
    for ax, (nombre, (valores, continua)) in zip(ejes.ravel(), colores.items()):
        if continua:
            sc = ax.scatter(*xy.T, c=valores, s=1.5, cmap="viridis", rasterized=True)
            fig.colorbar(sc, ax=ax, shrink=0.7)
        else:
            cats = np.unique(valores)
            for j, c in enumerate(cats):
                sel = valores == c
                ax.scatter(*xy[sel].T, s=1.5, color="#bbbbbb" if c < 0 else plt.cm.tab10(j % 10), label=str(c), rasterized=True)
            ax.legend(markerscale=6, fontsize=7, loc="best")
        ax.set_title(nombre, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in ejes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(titulo)
    figuras._guardar(fig, ruta)


def seccion_mapas(datos: dict, df: pl.DataFrame, etiquetas: dict, salida: Path, n_perm: int) -> str:
    texto = ["## Mapa de personas\n"]
    filas = {}
    for nombre, d in datos.items():
        p = d["personas"]
        vec, y, logit = p["test_vec"], p["test_y"], p.get("test_logit")
        et = alinear(p["test_id"], df, etiquetas)
        t0 = time.time()
        filas[nombre] = pureza_vecindades(vec, y, et, n_perm=n_perm)
        colores = {"clase real (y)": (y, False)}
        if logit is not None:
            colores["logit del modelo"] = (logit, True)
        colores |= {k: (v, False) for k, v in et.items()}
        for metodo, xy in proyecciones(vec).items():
            figura_mapa(xy, colores, salida / f"mapa_{nombre}_{metodo.lower()}.png", f"{MODELOS[nombre]}: {metodo}")
        print(f"mapas {nombre}: {time.time() - t0:.0f} s", flush=True)
    if filas:
        cols = list(next(iter(filas.values())))
        texto.append("Pureza de los 5 vecinos más cercanos (coseno, submuestra de 4.000 personas del test; entre paréntesis, el "
                     "valor esperado al azar). Si es mayor que el azar, personas parecidas en ese atributo quedan cerca.\n")
        texto.append("| modelo | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols))
        for nombre, f in filas.items():
            texto.append(f"| {MODELOS[nombre]} | " + " | ".join(f"{f[c]['pureza']:.3f} ({f[c]['nulo']:.3f})" for c in cols) + " |")
    return "\n".join(texto) + "\n"


# ---- TCAV -----------------------------------------------------------------------------------------------------------
def seccion_tcav(datos: dict, df: pl.DataFrame, grupos: dict, salida: Path) -> str:
    texto = ["## TCAV\n",
             "Puntaje = fracción de personas para las que mover el estado de todos sus tokens hacia el concepto sube la "
             "probabilidad de la clase 1 (0,5 = el concepto no importa; el nulo son direcciones al azar). `exactitud` = qué "
             "tan bien separa linealmente el concepto de las demás personas (datos retenidos). p con Bonferroni.\n"]
    for nombre in ("t1", "t3"):
        if nombre not in datos:
            continue
        p = datos[nombre]["personas"]
        gr_val = alinear(p["val_id"], df, grupos)
        for capa, (x, g) in {"última capa": ("m_ult", "g_ult"), "penúltima capa": ("m_pen", "g_pen")}.items():
            filas = tcav.tabla(p["val_" + x], p["test_" + g], p["test_y"], gr_val)
            texto.append(f"### {MODELOS[nombre]}, {capa}\n")
            texto.append("| concepto | clase real | TCAV | nulo | exactitud | p |\n|---|---|---|---|---|---|")
            for f in filas:
                marca = " *" if f["p"] < 0.05 else ""
                texto.append(f"| {f['concepto']} | {f['clase']} | {f['tcav']:.3f} ± {f['sd']:.3f} | {f['nulo']:.3f} | "
                             f"{f['exactitud']:.3f} | {f['p']:.1e}{marca} |")
            texto.append("")
    return "\n".join(texto)


# ---- saliency -------------------------------------------------------------------------------------------------------
def _categoria_de_token(c) -> np.ndarray:
    """categoria[token_id]; los especiales (y el [CLS], [SEP], etc.) quedan como «especial»."""
    t = c.tabla
    cat = np.full(int(t["id"].max()) + 1, "especial", dtype=object)
    cat[t["id"].to_numpy()] = t["categoria"].to_numpy()
    return cat


def agregados_saliency(s: dict, cat_token: np.ndarray) -> dict:
    """Por clase real: participación de cada categoría y de cada tramo de antigüedad del evento en el puntaje total, junto
    con su participación en la cantidad de tokens, y puntaje medio por token en eventos con y sin cambio de rama o tamaño."""
    persona = np.repeat(np.arange(len(s["largo"])), s["largo"])
    inicio = np.concatenate([[0], np.cumsum(s["largo"])])
    categoria = cat_token[s["token"]]
    meses = (dias_de_mes(np.array([CORTE]))[0] - s["abspos"]) / 30.4
    tramos = np.array([0, 3, 6, 12, 24, 60, 1e9])
    nombres_t = ["0-3", "3-6", "6-12", "12-24", "24-60", "60+"]
    es_evento = np.arange(len(persona)) - inicio[persona] >= N_FONDO
    out = {}
    for c in (0, 1):
        sel = (s["y"][persona] == c)
        tot_p, tot_n = s["puntaje"][sel].sum(), sel.sum()
        por_cat = {k: (s["puntaje"][sel & (categoria == k)].sum() / tot_p, (sel & (categoria == k)).sum() / tot_n)
                   for k in np.unique(categoria)}
        ev = sel & es_evento
        b = np.digitize(meses, tramos[1:-1])
        por_tramo = {nombres_t[i]: (s["puntaje"][ev & (b == i)].sum() / s["puntaje"][ev].sum(), (ev & (b == i)).sum() / ev.sum())
                     for i in range(len(nombres_t))}
        out[c] = {"categoria": por_cat, "meses": por_tramo, "cambio": _cambios(s, sel, persona, inicio, categoria, es_evento)}
    return out


def _cambios(s, sel, persona, inicio, categoria, es_evento) -> dict:
    """Puntaje medio por token en eventos donde cambia la empresa (rama o tamaño distintos del evento anterior) y en los demás.
    Un evento = tokens consecutivos con el mismo `abspos` dentro de una persona."""
    suma = {True: [0.0, 0], False: [0.0, 0]}
    for p in np.unique(persona[sel]):
        a, b = inicio[p] + N_FONDO, inicio[p + 1]
        pos, tok, cat, pun = s["abspos"][a:b], s["token"][a:b], categoria[a:b], s["puntaje"][a:b]
        cortes = np.flatnonzero(np.diff(pos) != 0) + 1
        previo = None
        for grupo in np.split(np.arange(len(pos)), cortes):
            firma = frozenset(tok[grupo][np.isin(cat[grupo], ["rama", "tamaño"])].tolist())
            cambio = previo is not None and firma != previo
            previo = firma
            suma[cambio][0] += pun[grupo].sum()
            suma[cambio][1] += len(grupo)
    return {("con cambio" if k else "sin cambio"): (v[0] / v[1] if v[1] else float("nan"), v[1]) for k, v in suma.items()}


def figura_historias(s: dict, cat_token: np.ndarray, ruta: Path, n: int = 4) -> None:
    """Línea de tiempo de algunas personas: un punto por token en (meses antes del corte, puntaje), coloreado por categoría."""
    inicio = np.concatenate([[0], np.cumsum(s["largo"])])
    rng = np.random.default_rng(0)
    elegidas = [rng.choice(np.flatnonzero(s["y"] == c), n // 2, replace=False) for c in (1, 0)]
    cats = sorted(set(cat_token[s["token"]]))
    color = {c: plt.cm.tab10(i % 10) for i, c in enumerate(cats)}
    fig, ejes = plt.subplots(2, n // 2, figsize=(5 * (n // 2), 7), sharey=True)
    for ax, k in zip(ejes.ravel(), np.concatenate(elegidas)):
        a, b = inicio[k] + N_FONDO, inicio[k + 1]
        meses = (dias_de_mes(np.array([CORTE]))[0] - s["abspos"][a:b]) / 30.4
        for c in cats:
            sel = cat_token[s["token"][a:b]] == c
            ax.scatter(-meses[sel], s["puntaje"][a:b][sel], s=10, color=color[c], label=c)
        ax.set_title(f"persona {int(s['id'][k])} (y = {int(s['y'][k])})", fontsize=9)
        ax.set_xlabel("meses respecto del corte")
    ejes[0, 0].set_ylabel("puntaje SmoothGrad")
    ejes[0, 0].legend(fontsize=7)
    figuras._guardar(fig, ruta)


def seccion_saliency(datos: dict, salida: Path) -> str:
    c = cargar_conceptos()
    cat = _categoria_de_token(c)
    texto = ["## Saliency (SmoothGrad)\n",
             "Participación en el puntaje total / participación en la cantidad de tokens. Un cociente mayor que 1 quiere decir "
             "que ese tipo de token pesa más de lo que le tocaría por su cantidad.\n"]
    for nombre in ("t1", "t3"):
        s = (datos.get(nombre) or {}).get("saliency")
        if s is None:
            continue
        ag = agregados_saliency(s, cat)
        figura_historias(s, cat, salida / f"historias_{nombre}.png")
        for cl in (1, 0):
            texto.append(f"### {MODELOS[nombre]}, personas con y = {cl}\n")
            texto.append("| categoría de token | % del puntaje | % de los tokens | cociente |\n|---|---|---|---|")
            for k, (a, b) in sorted(ag[cl]["categoria"].items(), key=lambda kv: -kv[1][0]):
                texto.append(f"| {k} | {100 * a:.1f} | {100 * b:.1f} | {a / b:.2f} |")
            texto.append("\n| meses antes del corte (eventos) | % del puntaje | % de los tokens | cociente |\n|---|---|---|---|")
            for k, (a, b) in ag[cl]["meses"].items():
                texto.append(f"| {k} | {100 * a:.1f} | {100 * b:.1f} | {a / b:.2f} |")
            texto.append("\n| evento | puntaje medio por token | tokens |\n|---|---|---|")
            for k, (a, n) in ag[cl]["cambio"].items():
                texto.append(f"| {k} | {a:.2e} | {n} |")
            texto.append("")
    return "\n".join(texto)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paso", choices=["conceptos", "analizar"])
    ap.add_argument("--entrada", default=str(ENTRADA))
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--solo", choices=["mapas", "tcav", "saliency"], help="sólo esa sección (para probar)")
    args = ap.parse_args(argv)
    if args.paso == "conceptos":
        df = construir_conceptos()
        print(f"{df.height} personas -> {CONCEPTOS}\n{df.describe()}")
        return
    df = pl.read_parquet(CONCEPTOS)
    etiquetas, grupos = conceptos_de(df)
    datos = cargar(Path(args.entrada))
    SALIDA.mkdir(parents=True, exist_ok=True)
    partes = ["# Resultados de la fase 7 (generado por `analisis.personas analizar`)\n"]
    if args.solo in (None, "mapas"):
        partes.append(seccion_mapas(datos, df, etiquetas, SALIDA, args.n_perm))
    if args.solo in (None, "tcav"):
        partes.append(seccion_tcav(datos, df, grupos, SALIDA))
    if args.solo in (None, "saliency"):
        partes.append(seccion_saliency(datos, SALIDA))
    destino = SALIDA / ("resultados.md" if args.solo is None else f"resultados_{args.solo}.md")
    destino.write_text("\n".join(partes))
    print(f"escrito {destino}")


if __name__ == "__main__":
    main()
