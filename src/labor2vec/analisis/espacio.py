"""Fase 9, paso local: qué información guarda el espacio de personas y qué tipologías salen de él (plans/PLAN_fase_9.md).

    python -m labor2vec.analisis.espacio proporciones          # cachea las proporciones de tokens de val y test (minutos)
    python -m labor2vec.analisis.espacio sondas [--n-boot 200]  # A1: sondas lineales sobre cada espacio
    python -m labor2vec.analisis.espacio ejes                   # A2: componentes principales del espacio pre-entrenado
    python -m labor2vec.analisis.espacio om                     # A3: distancias de optimal matching del test (~25 min, caché)
    python -m labor2vec.analisis.espacio tipologias [--k 8]     # A3: Ward en los tres espacios, perfiles y validez externa

Lee lo que dejó `analisis.extraer` en `runs/fase7/{t1,t3,pre}/personas.npz` y escribe las tablas en
`informes/fase9/resultados_<paso>.md`.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl

from labor2vec.analisis import personas
from labor2vec.tareas import tabulares, tiempo
from labor2vec.tokens.documento import Opciones
from labor2vec.tokens.vocabulario import Vocabulario
from labor2vec.utiles.rutas import CORPUS, INFORMES, RAIZ, T1, TAREAS

ENTRADA = RAIZ / "runs/fase7"
SALIDA = INFORMES / "fase9"
FIN_VENTANA = 239
MODELOS = {"pre": "pre-entrenado (vec)", "pre_pen": "pre-entrenado (penúltima capa)", "t1": "T1 ajustado", "t3": "T3 ajustado"}


# ---- insumos --------------------------------------------------------------------------------------------------------
def proporciones(split: str, corpus=CORPUS, ruta_t1=T1) -> tuple[np.ndarray, np.ndarray]:
    """(id, x) con las proporciones de tokens más sexo y edad de toda la cohorte de T1 del split (igual que `tabulares`)."""
    cache = TAREAS / f"prop_tokens_{split}_v{FIN_VENTANA}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["id"], z["x"]
    from labor2vec.tareas.datos import DatasetT1

    vocab = Vocabulario.cargar(Path(corpus) / "vocabulario.parquet", Path(corpus) / "ramas.parquet")
    op = Opciones(max_length=2560, fin_ventana=FIN_VENTANA, atributos="todos")
    ds = DatasetT1(Path(corpus) / f"eventos_{split}", vocab, op, False, ruta_t1, "y_2019")
    x = tabulares.atributos(ds, len(vocab))
    np.savez(cache, id=ds.id, x=x)
    return ds.id, x


def por_ids(ids_todos: np.ndarray, x: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Filas de `x` (indexada por `ids_todos`) en el orden de `ids`."""
    pos = {int(i): k for k, i in enumerate(ids_todos)}
    return x[[pos[int(i)] for i in ids]]


def objetivos(df: pl.DataFrame, ids: np.ndarray, split: str) -> tuple[dict, dict]:
    """(categóricos, continuos) de las personas `ids`: los conceptos de 2015 y las 20 variables de dinámica (con log1p,
    estandarizadas con la media y el desvío de val)."""
    etiquetas, _ = personas.conceptos_de(df)
    cat = personas.alinear(ids, df, etiquetas)
    cont = personas.alinear(ids, df, {"edad (años)": df["edad_corte"].to_numpy().astype(float),
                                      "salario (percentil 2015)": df["pct2015"].to_numpy().astype(float),
                                      "antigüedad (años)": df["antig_anios"].to_numpy().astype(float)})
    crudo = tabulares.variables_de_tiempo(split, ids, CORPUS, T1, FIN_VENTANA)
    cont |= {f"dinámica: {n}": crudo[:, k].astype(float) for k, n in enumerate(tiempo.NOMBRES)}
    return cat, cont


def dinamica_estandarizada(cont_val: dict, cont_test: dict) -> tuple[dict, dict]:
    """Aplica el log1p de `tiempo.transformar` a las variables de dinámica y estandariza con val; el resto queda igual."""
    nombres = [f"dinámica: {n}" for n in tiempo.NOMBRES]
    xv = np.stack([cont_val[n] for n in nombres], 1)
    xt = np.stack([cont_test[n] for n in nombres], 1)
    zv, media, desvio = tiempo.transformar(xv)
    zt = tiempo.transformar(xt, media, desvio)[0]
    for k, n in enumerate(nombres):
        cont_val[n], cont_test[n] = zv[:, k].astype(float), zt[:, k].astype(float)
    return cont_val, cont_test


# ---- A1: sondas -----------------------------------------------------------------------------------------------------
def _estandarizar(xv: np.ndarray, xt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    media, desvio = xv.mean(0), np.maximum(xv.std(0), 1e-6)
    return (xv - media) / desvio, (xt - media) / desvio


def _ic_bootstrap(medida, n: int, n_boot: int, rng) -> tuple[float, float]:
    vals = [medida(rng.integers(0, n, n)) for _ in range(n_boot)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def sonda_continua(xv, yv, xt, yt, n_boot: int = 200, semilla: int = 0) -> dict:
    """Ridge (alfa por validación cruzada en val) → R² en test con IC bootstrap. Las filas con dato faltante se descartan."""
    from sklearn.linear_model import RidgeCV

    okv, okt = ~np.isnan(yv), ~np.isnan(yt)
    if yt[okt].std() < 1e-6:  # variable constante en test (p. ej. meses desde el último registro): no hay nada que predecir
        return {"valor": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": int(okt.sum())}
    zv, zt = _estandarizar(xv[okv], xt[okt])
    m = RidgeCV(alphas=np.logspace(-1, 4, 6)).fit(zv, yv[okv])
    p, y = m.predict(zt), yt[okt]

    def r2(idx):
        return 1 - ((y[idx] - p[idx]) ** 2).sum() / max(((y[idx] - y[idx].mean()) ** 2).sum(), 1e-12)

    lo, hi = _ic_bootstrap(r2, len(y), n_boot, np.random.default_rng(semilla))
    return {"valor": float(r2(np.arange(len(y)))), "lo": lo, "hi": hi, "n": int(len(y))}


def sonda_categorica(xv, yv, xt, yt, n_boot: int = 200, semilla: int = 0, max_val: int = 8000) -> dict:
    """Logística multinomial con clases balanceadas (C elegido en una porción de val) → exactitud balanceada en test."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score

    rng = np.random.default_rng(semilla)
    okv, okt = yv >= 0, yt >= 0
    xv, yv, xt, yt = xv[okv], yv[okv], xt[okt], yt[okt]
    if len(yv) > max_val:
        s = rng.choice(len(yv), max_val, replace=False)
        xv, yv = xv[s], yv[s]
    zv, zt = _estandarizar(xv, xt)
    corte = int(0.7 * len(yv))
    mejor, c_mejor = -1.0, 0.01
    for c in (0.001, 0.01, 0.1):
        m = LogisticRegression(C=c, class_weight="balanced", max_iter=300).fit(zv[:corte], yv[:corte])
        b = balanced_accuracy_score(yv[corte:], m.predict(zv[corte:]))
        if b > mejor:
            mejor, c_mejor = b, c
    m = LogisticRegression(C=c_mejor, class_weight="balanced", max_iter=300).fit(zv, yv)
    p = m.predict(zt)

    def exactitud(idx):
        return balanced_accuracy_score(yt[idx], p[idx])

    lo, hi = _ic_bootstrap(exactitud, len(yt), n_boot, rng)
    return {"valor": float(exactitud(np.arange(len(yt)))), "lo": lo, "hi": hi, "n": int(len(yt)),
            "azar": float(1 / len(np.unique(yt)))}


def espacios_de(entrada: Path) -> dict:
    """{nombre: {split: (ids, X)}} para cada espacio disponible, más las proporciones de tokens recortadas a cada cohorte."""
    out = {}
    for nombre, (modelo, campo) in {"pre": ("pre", "vec"), "pre_pen": ("pre", "m_pen"), "t1": ("t1", "vec"),
                                    "t3": ("t3", "vec")}.items():
        ruta = Path(entrada) / modelo / "personas.npz"
        if not ruta.exists():
            continue
        z = np.load(ruta)
        out[nombre] = {s: (z[f"{s}_id"], z[f"{s}_{campo}"]) for s in ("val", "test")}
    return out


def seccion_sondas(entrada: Path, n_boot: int) -> str:
    df = pl.read_parquet(personas.CONCEPTOS)
    prop = {s: proporciones(s) for s in ("val", "test")}
    base = espacios_de(entrada)
    espacios: dict[str, dict] = {}
    for nombre, d in base.items():
        espacios[nombre] = d
        espacios[f"tokens (misma cohorte que {nombre})"] = {s: (d[s][0], por_ids(*prop[s], d[s][0])) for s in d}
        espacios[f"tokens + {nombre}"] = {s: (d[s][0], np.concatenate([por_ids(*prop[s], d[s][0]), d[s][1]], 1)) for s in d}
    filas: dict[str, dict] = {}
    t0 = time.time()
    for nombre, d in espacios.items():
        (idv, xv), (idt, xt) = d["val"], d["test"]
        cv, wv = objetivos(df, idv, "val")
        ct, wt = objetivos(df, idt, "test")
        wv, wt = dinamica_estandarizada(wv, wt)
        filas[nombre] = {}
        for k in cv:
            filas[nombre][k] = sonda_categorica(xv, cv[k], xt, ct[k], n_boot)
        for k in wv:
            filas[nombre][k] = sonda_continua(xv, wv[k], xt, wt[k], n_boot)
        print(f"sondas {nombre}: {time.time() - t0:.0f} s", flush=True)

    objs = list(next(iter(filas.values())))
    texto = ["## Sondas lineales\n",
             "Cada celda es el desempeño en test de una regresión lineal entrenada en val sobre el espacio de la fila: "
             "**exactitud balanceada** para las variables categóricas (entre paréntesis, lo esperable al azar) y **R²** para "
             "las continuas (edad, salario, antigüedad y las 20 variables de dinámica). Entre corchetes, el IC al 95 % por "
             "bootstrap del test.\n"]
    for titulo, sel in (("Categóricas (exactitud balanceada)", [o for o in objs if not o.startswith("dinámica") and "(" not in o]),
                        ("Continuas (R²)", [o for o in objs if "(" in o or o.startswith("dinámica")])):
        texto.append(f"### {titulo}\n")
        texto.append("| espacio | " + " | ".join(sel) + " |\n|---|" + "---|" * len(sel))
        for nombre, f in filas.items():
            celdas = []
            for o in sel:
                r = f[o]
                extra = f" ({r['azar']:.2f})" if "azar" in r else ""
                celdas.append(f"{r['valor']:.3f}{extra} [{r['lo']:.3f}, {r['hi']:.3f}]")
            texto.append(f"| {nombre} | " + " | ".join(celdas) + " |")
        texto.append("")
    return "\n".join(texto)


# ---- A2: ejes -------------------------------------------------------------------------------------------------------
def razon_de_correlacion(x: np.ndarray, grupos: np.ndarray) -> float:
    """η²: parte de la varianza de `x` que explican los grupos (grupos < 0 se descartan)."""
    ok = grupos >= 0
    if not ok.any():
        return 0.0
    x, g = x[ok], grupos[ok]
    total = ((x - x.mean()) ** 2).sum()
    if total == 0:
        return 0.0
    entre = sum(((g == a).sum()) * (x[g == a].mean() - x.mean()) ** 2 for a in np.unique(g))
    return float(entre / total)


def seccion_ejes(entrada: Path, n_comp: int = 10) -> str:
    df = pl.read_parquet(personas.CONCEPTOS)
    z = np.load(Path(entrada) / "pre" / "personas.npz")
    ids, x = z["test_id"], z["test_vec"].astype(np.float64)
    cat, cont = objetivos(df, ids, "test")
    zt = tiempo.transformar(np.stack([cont[f"dinámica: {n}"] for n in tiempo.NOMBRES], 1))[0]
    cont |= {f"dinámica: {n}": zt[:, k].astype(float) for k, n in enumerate(tiempo.NOMBRES)}
    xc = x - x.mean(0)
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    comp = xc @ vt[:n_comp].T
    var = s[:n_comp] ** 2 / (s ** 2).sum()
    texto = ["## Ejes del espacio pre-entrenado (componentes principales del test)\n",
             "Por componente: varianza explicada, las tres variables continuas con mayor |correlación| y los conceptos "
             "categóricos con mayor η² (parte de la varianza del componente que explica el concepto).\n",
             "| componente | varianza | continuas (correlación) | categóricos (η²) |\n|---|---|---|---|"]
    for c in range(n_comp):
        cors = {}
        for k, v in cont.items():
            ok = ~np.isnan(v)
            cors[k] = float(np.corrcoef(comp[ok, c], v[ok])[0, 1])
        top = sorted(cors, key=lambda k: -abs(cors[k]))[:3]
        etas = {k: razon_de_correlacion(comp[:, c], v) for k, v in cat.items()}
        top_e = sorted(etas, key=lambda k: -etas[k])[:2]
        texto.append(f"| PC{c + 1} | {100 * var[c]:.1f} % | " + "; ".join(f"{k} {cors[k]:+.2f}" for k in top) + " | "
                     + "; ".join(f"{k} {etas[k]:.2f}" for k in top_e) + " |")
    texto.append(f"\nVarianza acumulada de los {n_comp} primeros: {100 * var.sum():.1f} %.\n")
    return "\n".join(texto)


# ---- A3: tipologías de trayectorias ---------------------------------------------------------------------------------
# Toda la cohorte de test de T1 (la de `runs/fase7/pre`), en tres espacios con el mismo procedimiento: Ward sobre las
# distancias (coseno = euclídea sobre vectores normalizados, en `pre` y `tokens`; optimal matching, en `om`) y cortes del
# mismo árbol para k = 4..12.
KS = tuple(range(4, 13))
PALETA = {0: "#ffffff",  # sin registro; después salario (bajo, medio, alto) × tamaño (chica, mediana, grande)
          1: "#c6dbef", 2: "#6baed6", 3: "#2171b5", 4: "#c7e9c0", 5: "#74c476", 6: "#238b45",
          7: "#fdd0a2", 8: "#fd8d3c", 9: "#a63603"}
METODOS = {"pre": "espacio del modelo (pre-entrenado)", "tokens": "proporciones de tokens", "om": "optimal matching"}


def ids_test(entrada: Path) -> np.ndarray:
    return np.load(Path(entrada) / "pre" / "personas.npz")["test_id"]


def secuencias_test(ids: np.ndarray) -> np.ndarray:
    """(n, 120) estados mensuales 2006-01 a 2015-12 de `ids`, con caché."""
    from labor2vec.analisis import secuencias

    cache = TAREAS / f"secuencias_test_v{FIN_VENTANA}.npz"
    if cache.exists():
        z = np.load(cache)
        if np.array_equal(z["id"], ids):
            return z["s"]
    s = secuencias.estados_mensuales(ids)
    np.savez(cache, id=ids, s=s)
    return s


def distancias_om_test(ids: np.ndarray) -> np.ndarray:
    """Distancias de optimal matching entre todos los pares del test (formato condensado, float32), con caché en disco."""
    from labor2vec.analisis import secuencias

    cache = TAREAS / f"om_test_v{FIN_VENTANA}.npy"
    if cache.exists():
        return np.load(cache, mmap_mode="r")
    s = secuencias_test(ids)
    sub = secuencias.costos_trate(s)
    t0 = time.time()
    d = secuencias.distancias_om(s, sub)
    np.save(cache, d)
    print(f"OM: {len(s)} personas, {len(d):,} pares en {time.time() - t0:.0f} s", flush=True)
    return d


def _normalizar(x: np.ndarray, centrar: bool = False) -> np.ndarray:
    """Filas de norma 1 (distancia euclídea = coseno). Con `centrar`, se resta antes la media: los vectores comparten una
    componente común grande que, sin centrar, aplasta las distancias y borronea los grupos."""
    x = x.astype(np.float64)
    if centrar:
        x = x - x.mean(0)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def vectores_de(nombre: str, entrada: Path, ids: np.ndarray) -> np.ndarray:
    """Vectores normalizados (coseno) de `pre` o `tokens` en el orden de `ids`."""
    if nombre == "pre":
        z = np.load(Path(entrada) / "pre" / "personas.npz")
        return _normalizar(por_ids(z["test_id"], z["test_vec"], ids), centrar=True)
    return _normalizar(por_ids(*proporciones("test"), ids), centrar=True)


def arbol(nombre: str, entrada: Path, ids: np.ndarray) -> np.ndarray:
    """Matriz de enlace de Ward del método `nombre`, con caché."""
    from scipy.cluster.hierarchy import linkage

    cache = TAREAS / f"ward_{nombre}{"" if nombre == "om" else "_c"}_test_v{FIN_VENTANA}.npy"
    if cache.exists():
        return np.load(cache)
    t0 = time.time()
    if nombre == "om":
        z = linkage(np.asarray(distancias_om_test(ids), dtype=np.float64), method="ward")
    else:
        z = linkage(vectores_de(nombre, entrada, ids), method="ward")
    np.save(cache, z)
    print(f"Ward {nombre}: {time.time() - t0:.0f} s", flush=True)
    return z


def cortar(z: np.ndarray, k: int) -> np.ndarray:
    from scipy.cluster.hierarchy import fcluster

    return fcluster(z, k, criterion="maxclust") - 1


def submatriz(d: np.ndarray, n: int, idx: np.ndarray) -> np.ndarray:
    """Matriz cuadrada de las distancias entre las personas `idx` a partir del formato condensado de `n` personas."""
    idx = np.sort(np.asarray(idx))
    i, j = np.meshgrid(idx, idx, indexing="ij")
    a, b = np.minimum(i, j), np.maximum(i, j)
    pos = a * n - a * (a + 1) // 2 + b - a - 1
    out = np.asarray(d[np.where(a == b, 0, pos)], dtype=np.float32)
    out[a == b] = 0
    return out


def silueta(nombre: str, etiquetas: np.ndarray, entrada: Path, ids: np.ndarray, n_muestra: int, semilla: int = 0) -> float:
    """Silueta media sobre una muestra de `n_muestra` personas (la matriz completa de 18 mil no entra cómoda en memoria)."""
    from sklearn.metrics import silhouette_score

    s = np.sort(np.random.default_rng(semilla).choice(len(ids), min(n_muestra, len(ids)), replace=False))
    if len(np.unique(etiquetas[s])) < 2:
        return float("nan")
    if nombre == "om":
        return float(silhouette_score(submatriz(distancias_om_test(ids), len(ids), s), etiquetas[s], metric="precomputed"))
    return float(silhouette_score(vectores_de(nombre, entrada, ids)[s], etiquetas[s]))


def perfil_de_tipos(lab: np.ndarray, seqs: np.ndarray, cont: dict) -> list[dict]:
    """Por tipo: tamaño y promedios de algunas variables estáticas, de dinámica y de la secuencia mensual."""
    filas = []
    con_reg = seqs > 0
    nivel_sal = np.where(con_reg, (seqs.astype(int) - 1) // 3, np.nan)
    tam = np.where(con_reg, (seqs.astype(int) - 1) % 3, np.nan)
    for t in np.unique(lab):
        m = lab == t
        filas.append({"tipo": int(t), "n": int(m.sum()), "% del test": 100 * m.mean(),
                      "edad": np.nanmean(cont["edad (años)"][m]), "salario pct": np.nanmean(cont["salario (percentil 2015)"][m]),
                      "antigüedad": np.nanmean(cont["antigüedad (años)"][m]),
                      "cambios empresa 60m": np.nanmean(cont["dinámica: cambios_empresa_60"][m]),
                      "huecos 60m": np.nanmean(cont["dinámica: huecos_60"][m]),
                      "% meses sin registro": 100 * (~con_reg[m]).mean(),
                      "nivel salarial medio (0-2)": np.nanmean(nivel_sal[m]),
                      "% meses en empresa grande": 100 * np.nanmean(tam[m] == 2)})
    return filas


def figura_index(seqs: np.ndarray, lab: np.ndarray, ruta: Path, titulo: str, n_por_tipo: int = 120, semilla: int = 0) -> None:
    """Index plot: un panel por tipo, una fila por persona (muestra), un píxel por mes (2006-01 a 2015-12)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    rng = np.random.default_rng(semilla)
    tipos = np.unique(lab)
    cols = min(4, len(tipos))
    fig, ejes = plt.subplots(-(-len(tipos) // cols), cols, figsize=(3.6 * cols, 3.4 * -(-len(tipos) // cols)), squeeze=False)
    cmap = ListedColormap([PALETA[i] for i in range(10)])
    for ax, t in zip(ejes.ravel(), tipos):
        idx = np.flatnonzero(lab == t)
        idx = rng.choice(idx, min(n_por_tipo, len(idx)), replace=False)
        sub = seqs[idx]
        orden = np.lexsort((sub[:, -1], (sub > 0).sum(1)))  # por meses con registro y por el estado final
        ax.imshow(sub[orden], aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=9)
        ax.set_title(f"tipo {t} ({100 * np.mean(lab == t):.1f} %)", fontsize=9)
        ax.set_xticks([0, 60, 119])
        ax.set_xticklabels(["2006", "2011", "2015"], fontsize=7)
        ax.set_yticks([])
    for ax in ejes.ravel()[len(tipos):]:
        ax.axis("off")
    fig.suptitle(titulo)
    from labor2vec.analisis import figuras

    figuras._guardar(fig, ruta)


def validez_externa(lab_por_metodo: dict, ids: np.ndarray) -> list[dict]:
    """Cuánto predice el tipo los resultados de T1 a T4: información mutua normalizada y AUC por validación cruzada de una
    logística sobre las dummies del tipo (binarias: AUC; multiclase: AUC macro uno contra el resto)."""
    import pyarrow.parquet as pq
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import normalized_mutual_info_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    tareas = {"T1 (sale del empleo)": ("t1", "y_2019"), "T2 (quintil salarial 2019)": ("t2", "y"),
              "T3 (cambia de rama)": ("t3", "y"), "T4 (tamaño de empresa 2019)": ("t4", "y")}
    filas = []
    for nombre, (t, col) in tareas.items():
        tabla = pq.read_table(TAREAS / f"{t}.parquet", columns=["id_trabajador", col]).to_pydict()
        y_de = dict(zip(tabla["id_trabajador"], tabla[col]))
        en = np.array([int(i) in y_de for i in ids])
        y = np.array([y_de[int(i)] for i in ids[en]])
        for metodo, lab in lab_por_metodo.items():
            l = lab[en]
            d = (l[:, None] == np.unique(l)[None, :]).astype(float)
            p = cross_val_predict(LogisticRegression(max_iter=500), d, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                                  method="predict_proba")
            auc = roc_auc_score(y, p[:, 1]) if p.shape[1] == 2 else roc_auc_score(y, p, multi_class="ovr", average="macro")
            filas.append({"tarea": nombre, "método": metodo, "n": int(en.sum()), "NMI": float(normalized_mutual_info_score(y, l)),
                          "AUC": float(auc)})
    return filas


def seccion_tipologias(entrada: Path, k_perfil: int, n_sil: int) -> str:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    ids = ids_test(entrada)
    seqs = secuencias_test(ids)
    df = pl.read_parquet(personas.CONCEPTOS)
    _, cont = objetivos(df, ids, "test")
    arboles = {m: arbol(m, entrada, ids) for m in METODOS}
    part = {m: {k: cortar(z, k) for k in KS} for m, z in arboles.items()}
    texto = [f"## Tipologías de trayectorias ({len(ids):,} personas del test)\n",
             "Ward sobre distancias coseno (`pre`, `tokens`) o de optimal matching (`om`); mismos cortes del árbol para "
             "k = 4 a 12.\n"]

    texto.append(f"### Silueta (muestra de {n_sil} personas; más alta = grupos más compactos y separados)\n")
    texto.append("| k | " + " | ".join(METODOS) + " |\n|---|" + "---|" * len(METODOS))
    for k in KS:
        texto.append(f"| {k} | " + " | ".join(f"{silueta(m, part[m][k], entrada, ids, n_sil):.3f}" for m in METODOS) + " |")
    texto.append("")

    texto.append("### Acuerdo entre tipologías (ARI / información mutua normalizada)\n")
    pares = [("pre", "om"), ("pre", "tokens"), ("tokens", "om")]
    texto.append("| k | " + " | ".join(f"{a} contra {b}" for a, b in pares) + " |\n|---|" + "---|" * len(pares))
    for k in KS:
        celdas = [f"{adjusted_rand_score(part[a][k], part[b][k]):.3f} / {normalized_mutual_info_score(part[a][k], part[b][k]):.3f}"
                  for a, b in pares]
        texto.append(f"| {k} | " + " | ".join(celdas) + " |")
    texto.append("")

    texto.append(f"### Perfil de los tipos (k = {k_perfil})\n")
    for m, nombre in METODOS.items():
        lab = part[m][k_perfil]
        figura_index(seqs, lab, SALIDA / f"index_{m}_k{k_perfil}.png", f"{nombre}: k = {k_perfil}")
        filas = perfil_de_tipos(lab, seqs, cont)
        cols = [c for c in filas[0] if c != "tipo"]
        texto.append(f"**{nombre}**  (figura: `index_{m}_k{k_perfil}.png`)\n")
        texto.append("| tipo | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols))
        for f in filas:
            texto.append(f"| {f['tipo']} | " + " | ".join(f"{f[c]:.1f}" if isinstance(f[c], float) else str(f[c]) for c in cols) + " |")
        texto.append("")

    texto.append(f"### Validez externa (k = {k_perfil}): cuánto predice el tipo los resultados de 2016 a 2019\n")
    filas = validez_externa({m: part[m][k_perfil] for m in METODOS}, ids)
    texto.append("| tarea | método | n | NMI | AUC (validación cruzada) |\n|---|---|---|---|---|")
    for f in filas:
        texto.append(f"| {f['tarea']} | {f['método']} | {f['n']:,} | {f['NMI']:.4f} | {f['AUC']:.3f} |")
    return "\n".join(texto) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paso", choices=["proporciones", "sondas", "ejes", "om", "tipologias"])
    ap.add_argument("--entrada", default=str(ENTRADA))
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--k", type=int, default=8, help="número de tipos para los perfiles y la validez externa")
    ap.add_argument("--n-sil", type=int, default=5000, help="personas de la muestra para la silueta")
    args = ap.parse_args(argv)
    SALIDA.mkdir(parents=True, exist_ok=True)
    if args.paso == "proporciones":
        for s in ("val", "test"):
            t0 = time.time()
            _, x = proporciones(s)
            print(f"{s}: {x.shape} en {time.time() - t0:.0f} s", flush=True)
        return
    if args.paso == "om":
        d = distancias_om_test(ids_test(Path(args.entrada)))
        print(f"listo: {len(d):,} distancias")
        return
    if args.paso == "tipologias":
        texto = seccion_tipologias(Path(args.entrada), args.k, args.n_sil)
    else:
        texto = seccion_sondas(Path(args.entrada), args.n_boot) if args.paso == "sondas" else seccion_ejes(Path(args.entrada))
    destino = SALIDA / f"resultados_{args.paso}.md"
    destino.write_text(f"# Fase 9: {args.paso} (generado por `analisis.espacio`)\n\n" + texto)
    print(f"escrito {destino}")


if __name__ == "__main__":
    main()
