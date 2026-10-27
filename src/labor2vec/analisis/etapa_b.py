"""Fase 9, etapa B, paso local: análisis de lo que dejan los jobs de Mendieta (plans/PLAN_fase_9.md, sección 6).

    python -m labor2vec.analisis.etapa_b semillas         # B1: estabilidad entre semillas (CKA, vecinos, tipologías, sondas)
    python -m labor2vec.analisis.etapa_b cortes           # B2: trayectorias en el espacio (movimiento y transición entre tipos)
    python -m labor2vec.analisis.etapa_b intervenciones   # B3: efecto de cambiar una cosa de la historia
    python -m labor2vec.analisis.etapa_b controles        # tiempo contra orden: sondas sobre vectores con fechas u orden permutados
    python -m labor2vec.analisis.etapa_b extras           # encoder sin entrenar y sin «último valor» contra el pre-entrenado

Lee `runs/fase7/{pre,t1,t3}/personas.npz` (semilla 0, historia hasta 2015-12) y `runs/fase9/<nombre>/personas.npz`
(`scripts/cluster/fase9_encolar.sh`) y escribe `informes/fase9/resultados_<paso>.md`. Todo es sobre el test.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import adjusted_rand_score

from labor2vec.analisis import espacio, personas
from labor2vec.analisis import metricas as m
from labor2vec.tareas import tabulares, tiempo
from labor2vec.tareas.metricas import aul
from labor2vec.tokens.documento import INTERVENCIONES
from labor2vec.utiles.rutas import CORPUS, INFORMES, RAIZ, T1

ENTRADA7 = RAIZ / "runs/fase7"
ENTRADA9 = RAIZ / "runs/fase9"
SALIDA = INFORMES / "fase9"
FAMILIAS = {"pre": ("pre", ["pre_s1", "pre_s2"]), "t1": ("t1", ["t1_s1", "t1_s2"]), "t3": ("t3", ["t3_s1", "t3_s2"])}
CORTES = {179: "2010-12", 203: "2012-12", 227: "2014-12", 239: "2015-12"}
TOL = 1e-4  # diferencia máxima por coordenada por debajo de la cual el vector se considera el mismo
CLAVE_CAT = ["letra", "tamaño", "salario", "antigüedad"]
CLAVE_CONT = ["antigüedad (años)"] + [f"dinámica: {n}" for n in (
    "cambios_empresa_60", "meses_desde_cambio_empresa", "meses_desde_cambio_tamano", "cambios_tamano_60",
    "cambios_rama_60", "huecos_60", "hay_dif_36", "dif_salario_36")]


# ---- lectura y alineación -------------------------------------------------------------------------------------------
def cargar(ruta: Path) -> dict | None:
    """personas.npz de una corrida, con las claves sin prefijo de split (`test_id` → `id` en `["test"]`); None si falta."""
    ruta = Path(ruta) / "personas.npz"
    if not ruta.exists():
        return None
    z = np.load(ruta)
    out = {}
    for s in ("val", "test"):
        if f"{s}_id" in z.files:
            out[s] = {k[len(s) + 1:]: z[k] for k in z.files if k.startswith(s + "_")}
    return out


def alinear_a(ids_ref: np.ndarray, ids: np.ndarray, *arrays: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """(máscara sobre `ids_ref` de quienes están en `ids`, arreglos de `ids` reordenados para esas personas)."""
    pos = {int(i): k for k, i in enumerate(ids)}
    ok = np.array([int(i) in pos for i in ids_ref])
    k = np.array([pos[int(i)] for i in ids_ref[ok]], dtype=np.int64)
    return ok, [a[k] for a in arrays]


# ---- medidas puras ---------------------------------------------------------------------------------------------------
def cka_lineal(x: np.ndarray, y: np.ndarray) -> float:
    """CKA lineal entre dos representaciones de las mismas personas (no depende de la rotación ni de la escala de los ejes)."""
    x = x.astype(np.float64) - x.mean(0)
    y = y.astype(np.float64) - y.mean(0)
    return float(np.linalg.norm(x.T @ y) ** 2 / (np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)))


def solapamiento_vecinos(x: np.ndarray, y: np.ndarray, k: int = 10, n: int = 4000, semilla: int = 0) -> tuple[float, float]:
    """(fracción media de los k vecinos coseno de una persona que coinciden entre los dos espacios, valor esperado al azar),
    sobre una submuestra de `n` personas (los vecinos se buscan dentro de la submuestra)."""
    s = np.sort(np.random.default_rng(semilla).choice(len(x), min(n, len(x)), replace=False))
    a, b = m.vecinos(x[s], k), m.vecinos(y[s], k)
    comunes = (a[:, :, None] == b[:, None, :]).any(2).sum(1)
    return float(comunes.mean() / k), float(k / (len(s) - 1))


def tipologia(x: np.ndarray, k: int) -> np.ndarray:
    """Ward sobre vectores centrados y normalizados, cortado en k tipos (el mismo procedimiento que A3)."""
    from scipy.cluster.hierarchy import linkage

    return espacio.cortar(linkage(espacio._normalizar(x, centrar=True), method="ward"), k)


def matriz_transicion(antes: np.ndarray, despues: np.ndarray, k: int) -> np.ndarray:
    """Matriz (k, k) con la proporción de las personas del tipo i en `antes` que están en el tipo j en `despues`."""
    t = np.zeros((k, k))
    np.add.at(t, (antes, despues), 1)
    return t / np.maximum(t.sum(1, keepdims=True), 1)


def resumen_logit(base: np.ndarray, nuevo: np.ndarray, y: np.ndarray, afectada: np.ndarray) -> dict:
    """Efecto de una intervención sobre el logit: entre las afectadas, cambio medio, mediana, p5-p95 y fracción que sube; en
    todas, la AUL (contra `y`) antes y después, y la correlación de rangos."""
    from scipy.stats import spearmanr

    d = (nuevo - base)[afectada]
    return {"n_afectadas": int(afectada.sum()), "media": float(d.mean()) if len(d) else float("nan"),
            "mediana": float(np.median(d)) if len(d) else float("nan"),
            "p5": float(np.percentile(d, 5)) if len(d) else float("nan"),
            "p95": float(np.percentile(d, 95)) if len(d) else float("nan"),
            "sube": float((d > 0).mean()) if len(d) else float("nan"),
            "aul_antes": aul(y, base), "aul_despues": aul(y, nuevo), "spearman": float(spearmanr(base, nuevo)[0])}


def distancia_coseno_filas(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 1.0 - (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


# ---- sondas sobre un espacio -----------------------------------------------------------------------------------------
def sondear(val: dict, test: dict, df: pl.DataFrame, n_boot: int = 30) -> dict:
    """Valor de las sondas clave (A1) sobre el vector de una corrida: {objetivo: exactitud balanceada o R²}."""
    cv, wv = espacio.objetivos(df, val["id"], "val")
    ct, wt = espacio.objetivos(df, test["id"], "test")
    wv, wt = espacio.dinamica_estandarizada(wv, wt)
    out = {k: espacio.sonda_categorica(val["vec"], cv[k], test["vec"], ct[k], n_boot)["valor"] for k in CLAVE_CAT}
    out |= {k: espacio.sonda_continua(val["vec"], wv[k], test["vec"], wt[k], n_boot)["valor"] for k in CLAVE_CONT}
    return out


def _fila(nombre: str, valores: list[float], dec: int = 3) -> str:
    return f"| {nombre} | " + " | ".join(f"{v:.{dec}f}" for v in valores) + " |"


# ---- B1: semillas ----------------------------------------------------------------------------------------------------
def seccion_semillas(e7: Path, e9: Path, k: int = 8, n_boot: int = 30) -> str:
    df = pl.read_parquet(personas.CONCEPTOS)
    texto = ["## B1. Estabilidad entre semillas\n",
             "Cada semilla es un entrenamiento distinto (el pre-entrenado con otro azar; en `t1` y `t3`, sólo el ajuste fino con "
             "otro azar, a partir del mismo pre-entrenado). Sobre las mismas personas del test se mide: **CKA lineal** (similitud "
             "global de los espacios; 1 = idénticos salvo rotación o escala); **vecinos** (fracción de los 10 vecinos más "
             f"cercanos que coinciden; entre paréntesis, lo esperable al azar); **ARI** de las tipologías de k = {k} tipos "
             "armadas con cada espacio.\n"]
    sondas: dict[str, dict] = {}
    for fam, (base, extras) in FAMILIAS.items():
        corridas = {0: cargar(e7 / base)} | {i + 1: cargar(e9 / nombre) for i, nombre in enumerate(extras)}
        corridas = {s: c for s, c in corridas.items() if c is not None}
        if len(corridas) < 2:
            texto.append(f"### {fam}: sin datos suficientes (corridas disponibles: {sorted(corridas)})\n")
            continue
        ref = corridas[min(corridas)]["test"]["id"]
        vec = {}
        for s_, c in corridas.items():
            ok, (x,) = alinear_a(ref, c["test"]["id"], c["test"]["vec"])
            if ok.all():
                vec[s_] = x
        if len(vec) < 2:
            texto.append(f"### {fam}: las corridas no cubren las mismas personas del test\n")
            continue
        partes = {s_: tipologia(x, k) for s_, x in vec.items()}
        filas = []
        for a, b in [(a, b) for a in vec for b in vec if a < b]:
            sol, azar = solapamiento_vecinos(vec[a], vec[b])
            filas.append((f"{a} y {b}", cka_lineal(vec[a], vec[b]), sol, azar, float(adjusted_rand_score(partes[a], partes[b]))))
            print(f"semillas {fam} {a}-{b}: CKA {filas[-1][1]:.3f}", flush=True)
        texto.append(f"### {fam}\n\n| semillas | CKA | vecinos (azar) | ARI de tipologías |\n|---|---|---|---|")
        texto += [f"| {p} | {c:.3f} | {s:.3f} ({z:.4f}) | {r:.3f} |" for p, c, s, z, r in filas]
        texto.append("")
        for s, c in corridas.items():
            if "val" in c:
                t0 = time.time()
                sondas[f"{fam} semilla {s}"] = sondear(c["val"], c["test"], df, n_boot)
                print(f"sondas {fam} semilla {s}: {time.time() - t0:.0f} s", flush=True)
    if sondas:
        cols = CLAVE_CAT + CLAVE_CONT
        texto.append("### Sondas del vector por semilla (exactitud balanceada para las categóricas, R² para las continuas)\n")
        texto.append("| espacio | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols))
        texto += [_fila(n, [v[c] for c in cols]) for n, v in sondas.items()]
        texto.append("")
        for fam in FAMILIAS:
            vs = [v for n, v in sondas.items() if n.startswith(fam + " ")]
            if len(vs) >= 2:
                texto.append(f"Rango de las sondas en `{fam}` (máx − mín entre semillas): " + "; ".join(
                    f"{c} {max(v[c] for v in vs) - min(v[c] for v in vs):.3f}" for c in cols) + "\n")
    return "\n".join(texto) + "\n"


# ---- B2: cortes ------------------------------------------------------------------------------------------------------
def _vars_tiempo_en(ids: np.ndarray, fin: int) -> dict[str, np.ndarray]:
    crudo = tabulares.variables_de_tiempo("test", ids, CORPUS, T1, fin)
    return {n: crudo[:, k].astype(float) for k, n in enumerate(tiempo.NOMBRES)}


def seccion_cortes(e7: Path, e9: Path, k: int = 8) -> str:
    from sklearn.cluster import KMeans

    df = pl.read_parquet(personas.CONCEPTOS)
    base = cargar(e7 / "pre")["test"]
    ids = base["id"]
    vec = {239: base["vec"].astype(np.float64)}
    for corte in (179, 203, 227):
        c = cargar(e9 / f"pre_f{corte}")
        if c is not None:
            ok, (x,) = alinear_a(ids, c["test"]["id"], c["test"]["vec"])
            if ok.all():
                vec[corte] = x.astype(np.float64)
    if len(vec) < 2:
        return "## B2. Trayectorias en el espacio\n\nSin datos: faltan las corridas `pre_f179`, `pre_f203` y `pre_f227`.\n"

    pos = {int(i): j for j, i in enumerate(df["id_trabajador"].to_numpy())}
    antig = df["antig_anios"].to_numpy().astype(float)[[pos[int(i)] for i in ids]]
    primer_mes = 239 - np.rint(12 * antig)  # primer mes de la persona en el panel
    mu = vec[239].mean(0)

    def normal(x):
        return espacio._normalizar(x - mu)

    km = KMeans(k, n_init=5, random_state=0).fit(normal(vec[239]))
    tipo = {c: km.predict(normal(x)) for c, x in vec.items()}
    tv239 = _vars_tiempo_en(ids, 239)

    texto = ["## B2. Trayectorias en el espacio\n",
             "El mismo modelo pre-entrenado, con la historia de cada persona cortada en distintos meses (sólo ve lo anterior al "
             "corte). Sólo se cuentan las personas con al menos 12 meses de historia en el corte. Los tipos son los de un k-means "
             f"de k = {k} sobre los vectores de 2015-12 (centrados y normalizados); los vectores de los cortes anteriores se "
             "asignan al centroide más cercano con el mismo centrado. Los cortes más tempranos tienen menos historia, así que "
             "parte del movimiento es sólo falta de información: por eso se compara por estratos de historia.\n"]
    for corte in sorted(c for c in vec if c < 239):
        con = primer_mes <= corte - 11
        meses_hist = (corte - primer_mes)[con]
        distancia = distancia_coseno_filas(normal(vec[corte])[con], normal(vec[239])[con])
        texto.append(f"### Corte {CORTES[corte]} contra 2015-12 ({int(con.sum()):,} personas con historia)\n")
        texto.append(f"Distancia coseno entre el vector del corte y el de 2015-12: mediana {np.median(distancia):.3f}, "
                     f"p25 {np.percentile(distancia, 25):.3f}, p75 {np.percentile(distancia, 75):.3f}.\n")
        cortes_estrato = np.quantile(meses_hist, [1 / 3, 2 / 3])
        estrato = np.searchsorted(cortes_estrato, meses_hist, side="right")
        texto.append("| cambió entre el corte y 2015-12 | estrato de historia en el corte | n (cambió) | n (no cambió) | mediana (cambió) | "
                     "mediana (no cambió) | diferencia |\n|---|---|---|---|---|---|---|")
        for nombre, var in (("empleador", "meses_desde_cambio_empresa"), ("tamaño", "meses_desde_cambio_tamano"),
                            ("rama", "meses_desde_cambio_rama")):
            cambio = (tv239[var] < (239 - corte))[con]
            for e, rotulo in enumerate(("poca historia", "media", "mucha")):
                s = estrato == e
                a, b = distancia[s & cambio], distancia[s & ~cambio]
                if len(a) >= 20 and len(b) >= 20:
                    texto.append(f"| {nombre} | {rotulo} | {len(a):,} | {len(b):,} | {np.median(a):.3f} | {np.median(b):.3f} | "
                                 f"{np.median(a) - np.median(b):+.3f} |")
        texto.append("")
        tr = matriz_transicion(tipo[corte][con], tipo[239][con], k)
        texto.append(f"Transición entre tipos, de {CORTES[corte]} (filas) a 2015-12 (columnas); proporción de cada fila. "
                     f"Permanencia (diagonal ponderada): {float(np.average(np.diag(tr), weights=np.bincount(tipo[corte][con], minlength=k))):.3f}.\n")
        texto.append("| tipo | " + " | ".join(str(j) for j in range(k)) + " |\n|---|" + "---|" * k)
        texto += [f"| {i} | " + " | ".join(f"{tr[i, j]:.2f}" for j in range(k)) + " |" for i in range(k)]
        texto.append("")
    return "\n".join(texto) + "\n"


# ---- B3: intervenciones ----------------------------------------------------------------------------------------------
def seccion_intervenciones(e7: Path, e9: Path, k: int = 8) -> str:
    from sklearn.cluster import KMeans
    from sklearn.linear_model import Ridge

    df = pl.read_parquet(personas.CONCEPTOS)
    texto = ["## B3. Intervenciones sobre la historia\n",
             "Se cambia una cosa de la historia de cada persona del test y se vuelve a correr el modelo. «Afectadas» son las "
             "personas cuyo vector cambió (si la intervención no toca nada en su historia, queda igual). El cambio del logit es "
             "la predicción de la clase 1 menos la de la clase 0 (sólo en los modelos ajustados). Es lo que el modelo hace con "
             "la historia, no un efecto causal en el mundo real.\n",
             "- `sin_ultimo_anio`: se borran los eventos de los últimos 12 meses.\n"
             "- `sin_cambios_recientes`: la última relación conserva la rama y el tamaño de la anterior.\n"
             "- `salario_arriba` / `salario_abajo`: el percentil de los eventos del último año sube o baja 10 puntos.\n"
             "- `atrasar_2_anios`: todas las fechas, y con ellas las edades, pasan 24 meses atrás.\n"]
    for modelo in ("t1", "t3", "pre"):
        base = cargar(e7 / modelo)
        if base is None:
            continue
        b = base["test"]
        ids = b["id"]
        corridas = {i: cargar(e9 / f"{modelo}_{i}") for i in INTERVENCIONES}
        corridas = {i: c for i, c in corridas.items() if c is not None}
        if not corridas:
            texto.append(f"### {modelo}: sin corridas de intervención\n")
            continue
        # sonda de salario sobre el propio vector (para leer cuánto «mueve» el salario la intervención)
        ridge = None
        if "val" in base:
            _, cont = espacio.objetivos(df, base["val"]["id"], "val")
            y_sal = cont["salario (percentil 2015)"].astype(float)
            con_dato = ~np.isnan(y_sal)
            ridge = Ridge(alpha=100.0).fit(base["val"]["vec"][con_dato], y_sal[con_dato])
        km = KMeans(k, n_init=5, random_state=0).fit(espacio._normalizar(b["vec"], centrar=True)) if modelo == "pre" else None
        mu = b["vec"].astype(np.float64).mean(0)
        texto.append(f"### {modelo}\n")
        cols = ["afectadas", "Δ logit medio", "mediana", "p5 a p95", "% sube", "AUL antes → después", "Δ salario implícito (pts)", "cambia de tipo"]
        texto.append("| intervención | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols))
        for nombre, c in corridas.items():
            ok, (vec, logit) = alinear_a(ids, c["test"]["id"], c["test"]["vec"], c["test"].get("logit", np.zeros(len(c["test"]["id"]))))
            if not ok.all():
                continue
            afectada = np.abs(vec - b["vec"]).max(1) > TOL
            celdas = [f"{afectada.sum():,} ({100 * afectada.mean():.0f} %)"]
            if "logit" in b:
                r = resumen_logit(b["logit"], logit, b["y"], afectada)
                celdas += [f"{r['media']:+.3f}", f"{r['mediana']:+.3f}", f"{r['p5']:+.2f} a {r['p95']:+.2f}", f"{100 * r['sube']:.0f}",
                           f"{r['aul_antes']:.3f} → {r['aul_despues']:.3f}"]
            else:
                celdas += ["–"] * 5
            if ridge is not None and afectada.any():
                celdas.append(f"{float((ridge.predict(vec) - ridge.predict(b['vec']))[afectada].mean()):+.2f}")
            else:
                celdas.append("–")
            if km is not None:
                cambia = km.predict(espacio._normalizar(vec - mu)) != km.predict(espacio._normalizar(b["vec"] - mu))
                celdas.append(f"{100 * cambia.mean():.0f} % ({100 * cambia[afectada].mean():.0f} % de las afectadas)" if afectada.any() else "–")
            else:
                celdas.append("–")
            texto.append(f"| {nombre} | " + " | ".join(celdas) + " |")
        texto.append("")
    texto.append("«Δ salario implícito»: cambio medio, entre las afectadas, de lo que una sonda lineal (ridge, ajustada en val sobre "
                 "el propio vector) predice para el percentil salarial de 2015. Si el modelo registra el salario, `salario_arriba` "
                 "debería dar un valor positivo cercano a +10 y `salario_abajo` uno negativo.\n")
    return "\n".join(texto) + "\n"


# ---- tiempo contra orden ---------------------------------------------------------------------------------------------
def seccion_controles(e7: Path, e9: Path, n_boot: int = 30) -> str:
    df = pl.read_parquet(personas.CONCEPTOS)
    base = cargar(e7 / "pre")
    corridas = {"pre (sin tocar)": base, "fechas permutadas": cargar(e9 / "pre_ctrl_tiempo"), "orden permutado": cargar(e9 / "pre_ctrl_orden")}
    corridas = {n: c for n, c in corridas.items() if c is not None and "val" in c}
    if len(corridas) < 2:
        return "## Tiempo contra orden\n\nSin datos: faltan las corridas `pre_ctrl_tiempo` y `pre_ctrl_orden`.\n"
    cols = CLAVE_CAT + CLAVE_CONT
    texto = ["## Tiempo contra orden\n",
             "Sondas (A1) sobre el vector del pre-entrenado cuando las historias llegan al modelo con las **fechas y edades "
             "permutadas** entre los eventos de la persona (mismo orden, sin saber cuándo pasó cada uno) o con el **orden "
             "permutado** (cada evento conserva su fecha, pero cambia de lugar). Si una variable de dinámica cae sin fechas, el "
             "modelo la lee del tiempo; si cae con el orden permutado, depende del orden de los eventos.\n",
             "| entrada | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols)]
    for nombre, c in corridas.items():
        texto.append(_fila(nombre, [sondear(c["val"], c["test"], df, n_boot)[x] for x in cols]))
    return "\n".join(texto) + "\n"


def seccion_extras(e7: Path, e9: Path, n_boot: int = 30) -> str:
    df = pl.read_parquet(personas.CONCEPTOS)
    corridas = {"pre-entrenado": cargar(e7 / "pre"), "sin «último valor»": cargar(e9 / "pre_sinuv"), "pesos sin entrenar": cargar(e9 / "pre_azar")}
    corridas = {n: c for n, c in corridas.items() if c is not None and "val" in c}
    if len(corridas) < 2:
        return "## Controles del encoder\n\nSin datos: faltan las corridas `pre_sinuv` y `pre_azar`.\n"
    cols = CLAVE_CAT + CLAVE_CONT
    texto = ["## Controles del encoder\n",
             "Mismas sondas (A1) con tres encoders: el pre-entrenado, el pre-entrenado **sin rama, provincia ni antigüedad de la "
             "empresa** (`pre_150ep_sinuv`, que ve documentos sin esos atributos) y uno con **pesos sin entrenar** (el mismo "
             "diseño, sin ningún aprendizaje). Lo primero separa el efecto de los atributos de «último valor»; lo segundo, cuánto "
             "aporta el pre-entrenamiento.\n",
             "| encoder | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols)]
    for nombre, c in corridas.items():
        texto.append(_fila(nombre, [sondear(c["val"], c["test"], df, n_boot)[x] for x in cols]))
    return "\n".join(texto) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paso", choices=["semillas", "cortes", "intervenciones", "controles", "extras"])
    ap.add_argument("--entrada7", default=str(ENTRADA7))
    ap.add_argument("--entrada9", default=str(ENTRADA9))
    ap.add_argument("--k", type=int, default=8, help="número de tipos")
    ap.add_argument("--n-boot", type=int, default=30)
    args = ap.parse_args(argv)
    e7, e9 = Path(args.entrada7), Path(args.entrada9)
    SALIDA.mkdir(parents=True, exist_ok=True)
    texto = {"semillas": lambda: seccion_semillas(e7, e9, args.k, args.n_boot), "cortes": lambda: seccion_cortes(e7, e9, args.k),
             "intervenciones": lambda: seccion_intervenciones(e7, e9, args.k), "controles": lambda: seccion_controles(e7, e9, args.n_boot),
             "extras": lambda: seccion_extras(e7, e9, args.n_boot)}[args.paso]()
    destino = SALIDA / f"resultados_{args.paso}.md"
    destino.write_text(f"# Fase 9, etapa B: {args.paso} (generado por `analisis.etapa_b`)\n\n" + texto)
    print(f"escrito {destino}")


if __name__ == "__main__":
    main()
