"""Evaluación de T1-T4: tabla tipo A2 del paper con IC por bootstrap, subgrupos, calibración y figuras.

    python -m labor2vec.tareas.evaluar [--tarea t1|t2|t3|t4|t2c|t3d|t4c] [--prefijo t3_uv] [--runs runs] [--bootstrap 1000]

Busca las predicciones de test que escriben `ajustar` y `tabulares` (`predicciones_test.parquet`) en `runs/` según los
patrones de cada tarea y escribe en `informes/fase6/` (T1) o `informes/fase6b/<tarea>/` (T2-T4).
- T1 y T3 (binarias): tabla con AUL, AUC, PR-AUC y C-MCC; en T1 también la misma predicción contra `y_2021`; subgrupos,
  calibración (Brier) y curvas.
- T2 y T4 (multiclase ordenadas): QWK, MAE, F1 macro, exactitud balanceada y AUC uno-contra-resto, con la persistencia
  (T2: repetir el quintil de 2015; T4: «igual») como línea de base; subgrupos y matrices de confusión.
Los modelos con varias semillas se reportan como la media entre semillas, con el IC de esa media.
Fase 8 (plans/PLAN_fase_8.md): T2c (cambio de quintil), T3d (destino de la letra) y T4c (cambio de tamaño sí/no), con la
salida en `informes/fase8/<tarea>/`. `--prefijo` cambia el prefijo de las corridas (`t3_uv` busca `t3_uv_l2v_s*`,
`t3_uv_tab/…`) para evaluar una variante; `--runs runs/ctrl_tiempo` evalúa las predicciones con un control temporal.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from labor2vec.tareas import metricas
from labor2vec.utiles.rutas import INFORMES, RAIZ, T1
from labor2vec.utiles.rutas import TAREAS as DIR_TAREAS

# nombre en la tabla -> patrón de carpetas dentro de runs/ (cada una con predicciones_{val,test}.parquet)
MODELOS = {
    "Transformer pre-entrenado (L2V)": "t1_l2v_lr0001*",
    "Transformer sin pre-entrenar": "t1_scratch_s*",
    "GRU": "t1_gru_s*",
    "FFNN": "t1_tab/ffnn_s*",
    "Regresión logística": "t1_tab/logistica",
    "Tabla de vida (edad y sexo)": "t1_tab/tabla_vida",
}
METRICAS = [("c_mcc", "C-MCC"), ("aul", "AUL"), ("auc", "AUC"), ("pr_auc", "PR-AUC"), ("f1_corr", "F1 corr."),
            ("exactitud_corr", "Exactitud corr.")]
# Tabla A2 del paper (mortalidad en Dinamarca): sólo para comparar el orden y las distancias entre modelos
PAPER = {"Transformer pre-entrenado (L2V)": (0.413, 0.845), "GRU": (0.369, 0.834), "FFNN": (0.340, 0.822),
         "Regresión logística": (0.149, 0.735), "Tabla de vida (edad y sexo)": (0.059, 0.650)}


def _patrones(t: str) -> dict:
    return {"Transformer pre-entrenado (L2V)": f"{t}_l2v_s*", "Transformer sin pre-entrenar": f"{t}_scratch_s*",
            "GRU": f"{t}_gru_s*", "FFNN": f"{t}_tab/ffnn_s*", "Regresión logística": f"{t}_tab/logistica",
            "Logística con estado actual": f"{t}_tab/logistica_estado",
            "Tabla de vida (edad y sexo)": f"{t}_tab/tabla_vida"}


CONFIG = {
    "t1": dict(binaria=True, tabla=T1, patrones=MODELOS, etiquetas=[("y_2019", "tabla_y2019.md"), ("y_2021", "tabla_y2021.md")],
               salida=INFORMES / "fase6"),
    "t3": dict(binaria=True, tabla=DIR_TAREAS / "t3.parquet", patrones=_patrones("t3"), etiquetas=[("y", "tabla.md")],
               salida=INFORMES / "fase6b" / "t3"),
    "t2": dict(binaria=False, K=5, tabla=DIR_TAREAS / "t2.parquet", patrones=_patrones("t2"), monitor="qwk",
               metricas=[("qwk", "QWK"), ("mae", "MAE"), ("f1_macro", "F1 macro"), ("exactitud_bal", "Exact. balanc."),
                         ("auc_macro", "AUC macro")],
               estado="q2015", clases=["Q1", "Q2", "Q3", "Q4", "Q5"], salida=INFORMES / "fase6b" / "t2"),
    "t2c": dict(binaria=False, K=3, tabla=DIR_TAREAS / "t2c.parquet", patrones=_patrones("t2c"), monitor="auc_macro",
                metricas=[("auc_macro", "AUC macro"), ("qwk", "QWK"), ("mae", "MAE"), ("f1_macro", "F1 macro"),
                          ("exactitud_bal", "Exact. balanc.")],
                estado=None, clases=["baja", "igual", "sube"], salida=INFORMES / "fase8" / "t2c"),
    "t3d": dict(binaria=False, K=6, tabla=DIR_TAREAS / "t3d.parquet", patrones=_patrones("t3d"), monitor="auc_macro",
                metricas=[("auc_macro", "AUC macro"), ("exactitud_bal", "Exact. balanc."), ("f1_macro", "F1 macro")],
                estado=None, sin_persistencia=True, clases=[f"dest. {k}" for k in range(1, 6)] + ["otras"],
                salida=INFORMES / "fase8" / "t3d"),
    "t4c": dict(binaria=True, tabla=DIR_TAREAS / "t4c.parquet", patrones=_patrones("t4c"), etiquetas=[("y", "tabla.md")],
                salida=INFORMES / "fase8" / "t4c"),
    "t4": dict(binaria=False, K=3, tabla=DIR_TAREAS / "t4.parquet", patrones=_patrones("t4"), monitor="auc_macro",
               metricas=[("auc_macro", "AUC macro"), ("qwk", "QWK"), ("exactitud_bal", "Exact. balanc."),
                         ("f1_macro", "F1 macro"), ("auc_c0", "AUC más chica"), ("auc_c2", "AUC más grande")],
               estado=None, clases=["más chica", "igual", "más grande"], salida=INFORMES / "fase6b" / "t4"),
}


def leer(carpeta: Path, split: str = "test") -> dict:
    t = pq.read_table(carpeta / f"predicciones_{split}.parquet").to_pydict()
    orden = np.argsort(t["id_trabajador"])
    d = {k: np.asarray(v)[orden] for k, v in t.items()}
    k = 0
    while f"p_{k}" in d:
        k += 1
    if k:
        d["P"] = np.column_stack([d[f"p_{c}"] for c in range(k)])  # matriz (n, K) de probabilidades
    return d


def cargar_modelos(runs: Path, patrones: dict = MODELOS) -> dict:
    """{nombre: [predicciones por semilla]} de lo que haya en `runs`; los ids de todas las corridas deben coincidir."""
    out = {}
    for nombre, patron in patrones.items():
        carpetas = sorted(p for p in runs.glob(patron) if (p / "predicciones_test.parquet").exists())
        if carpetas:
            out[nombre] = [leer(c) for c in carpetas]
    ids = {n: tuple(v[0]["id_trabajador"]) for n, v in out.items()}
    assert len(set(ids.values())) <= 1, "los modelos no se evaluaron sobre los mismos trabajadores"
    return out


def formato(m: tuple, decimales: int = 3) -> str:
    return f"{m[0]:.{decimales}f} [{m[1]:.{decimales}f}, {m[2]:.{decimales}f}]"


def _columna(ruta, columna: str, ids) -> np.ndarray:
    """Valores de una columna de la tabla de la tarea, alineados con `ids`."""
    t = pq.read_table(ruta, columns=["id_trabajador", columna]).to_pydict()
    fila = {i: k for k, i in enumerate(t["id_trabajador"])}
    return np.array([t[columna][fila[int(i)]] for i in ids])


# ---- tareas binarias (T1 y T3) ---------------------------------------------------------------------------------
def tabla(modelos: dict, etiqueta: str, n_boot: int, referencia: dict) -> tuple[str, dict]:
    y = next(iter(modelos.values()))[0][etiqueta].astype(bool)
    azar = np.random.default_rng(20261020).random(len(y))
    todo = {**modelos, "Azar": [{"p": azar}], "Mayoría": [{"p": np.zeros(len(y))}]}
    filas, res = [], {}
    for nombre, preds in todo.items():
        r = metricas.bootstrap(y, [p["p"] for p in preds], n_boot, semilla=1)
        res[nombre] = {"n_semillas": len(preds), **{k: v for k, v in r.items()}}
        if len(preds) > 1:
            por_semilla = [metricas.todas(y, p["p"]) for p in preds]
            res[nombre]["sd_aul"] = float(np.std([s["aul"] for s in por_semilla], ddof=1))
            res[nombre]["sd_c_mcc"] = float(np.std([s["c_mcc"] for s in por_semilla], ddof=1))
        filas.append(f"| {nombre} | {len(preds)} | " + " | ".join(formato(r[k]) for k, _ in METRICAS) + " |")
    cab = "| Modelo | Semillas | " + " | ".join(n for _, n in METRICAS) + " |\n|---|---|" + "---|" * len(METRICAS)
    return f"Etiqueta `{etiqueta}`, test n = {len(y):,}, prevalencia {y.mean():.3f}, umbral 0,5.\n\n" + cab + "\n" + "\n".join(filas), res


def _grupos(modelos: dict, ruta_t1) -> dict:
    ids = next(iter(modelos.values()))[0]["id_trabajador"]
    sexo, edad = _columna(ruta_t1, "sexo", ids), _columna(ruta_t1, "edad_corte", ids)
    return {"mujeres": sexo == 1, "varones": sexo == 2, "25-34": (edad >= 25) & (edad <= 34),
            "35-44": (edad >= 35) & (edad <= 44), "45-55": (edad >= 45)}


def subgrupos(modelos: dict, ruta_t1=T1, etiqueta: str = "y_2019") -> str:
    grupos = _grupos(modelos, ruta_t1)
    y = next(iter(modelos.values()))[0][etiqueta].astype(bool)
    cab = "| Modelo | " + " | ".join(f"{g} (n={m.sum():,}, tasa {y[m].mean():.3f})" for g, m in grupos.items()) + " |"
    filas = [f"| {n} | " + " | ".join(f"{np.mean([metricas.aul(y[m], p['p'][m]) for p in ps]):.3f}" for m in grupos.values()) + " |"
             for n, ps in modelos.items()]
    return f"AUL por subgrupo (media entre semillas), etiqueta `{etiqueta}`.\n\n" + cab + "\n|---|" + "---|" * len(grupos) + "\n" + "\n".join(filas)


def calibracion(modelos: dict, etiqueta: str = "y_2019") -> str:
    y = next(iter(modelos.values()))[0][etiqueta].astype(bool)
    prev = y.mean()
    filas = []
    for n, ps in modelos.items():
        pc = [metricas.corregir_prevalencia(p["p"], prev) for p in ps]
        filas.append(f"| {n} | {np.mean([metricas.brier(y, p) for p in pc]):.4f} |")
    base = metricas.brier(y, np.full(len(y), prev))
    return (f"Brier con las probabilidades llevadas a la prevalencia real ({prev:.3f}); constante = {base:.4f}.\n\n"
            "| Modelo | Brier |\n|---|---|\n" + "\n".join(filas))


def figuras(modelos: dict, salida: Path, etiqueta: str = "y_2019") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = next(iter(modelos.values()))[0][etiqueta].astype(bool)
    prev = y.mean()
    fig, ejes = plt.subplots(1, 3, figsize=(15, 4.5))
    for n, ps in modelos.items():
        p = ps[0]["p"]
        orden = np.argsort(-p, kind="mergesort")
        frac = np.arange(1, len(y) + 1) / len(y)
        ejes[0].plot(frac, np.cumsum(y[orden]) / y.sum(), label=n)           # curva de lift (ganancia acumulada)
        ejes[1].plot(*_roc(y, p), label=n)
        cal = metricas.calibracion(y, metricas.corregir_prevalencia(p, prev), 10)
        ejes[2].plot(cal[:, 0], cal[:, 1], "o-", label=n)
    ejes[0].plot([0, 1], [0, 1], "k:")
    ejes[0].set(title="Ganancia acumulada (AUL)", xlabel="fracción de trabajadores con mayor riesgo", ylabel="fracción de positivos")
    ejes[1].plot([0, 1], [0, 1], "k:")
    ejes[1].set(title="Curva ROC", xlabel="falsos positivos", ylabel="verdaderos positivos")
    tope = max(0.6, 1.2 * prev)
    ejes[2].plot([0, tope], [0, tope], "k:")
    ejes[2].set(title="Calibración (prevalencia corregida)", xlabel="probabilidad predicha", ylabel="frecuencia observada")
    ejes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(salida / "curvas.png", dpi=110)


def _roc(y, p):
    orden = np.argsort(-p, kind="mergesort")
    ys = y[orden]
    return np.concatenate([[0], np.cumsum(~ys) / (~ys).sum()]), np.concatenate([[0], np.cumsum(ys) / ys.sum()])


# ---- tareas multiclase ordenadas (T2 y T4) ---------------------------------------------------------------------
def _uno_hot(clases: np.ndarray, k: int) -> np.ndarray:
    return (np.asarray(clases, dtype=int)[:, None] == np.arange(k)[None, :]).astype(float)


def _con_base(modelos: dict, y: np.ndarray, cfg: dict) -> dict:
    """Agrega persistencia, mayoría y azar como «modelos» de una sola corrida con matrices de probabilidad."""
    k, n = cfg["K"], len(y)
    ids = next(iter(modelos.values()))[0]["id_trabajador"]
    if cfg["estado"]:
        persistencia = _uno_hot(_columna(cfg["tabla"], cfg["estado"], ids), k)
    else:
        persistencia = _uno_hot(np.full(n, 1), k)  # T4: «igual»
    azar = np.random.default_rng(20261020).dirichlet(np.ones(k), n)
    mayoria = _uno_hot(np.full(n, np.bincount(y, minlength=k).argmax()), k)
    base = {} if cfg.get("sin_persistencia") else {"Persistencia": [{"P": persistencia}]}  # T3d es nominal: no hay «igual»
    return {**modelos, **base, "Mayoría": [{"P": mayoria}], "Azar": [{"P": azar}]}


def tabla_multi(modelos: dict, cfg: dict, n_boot: int) -> tuple[str, dict]:
    y = next(iter(modelos.values()))[0]["y"].astype(int)
    todo = _con_base(modelos, y, cfg)
    filas, res = [], {}
    for nombre, preds in todo.items():
        r = metricas.bootstrap_multi(y, [p["P"] for p in preds], n_boot, semilla=1)
        res[nombre] = {"n_semillas": len(preds), **r}
        if len(preds) > 1:
            res[nombre]["sd"] = {k: float(np.std([metricas.multiclase(y, p["P"])[k] for p in preds], ddof=1))
                                 for k, _ in cfg["metricas"]}
        filas.append(f"| {nombre} | {len(preds)} | " + " | ".join(formato(r[k]) for k, _ in cfg["metricas"]) + " |")
    cab = "| Modelo | Semillas | " + " | ".join(n for _, n in cfg["metricas"]) + " |\n|---|---|" + "---|" * len(cfg["metricas"])
    dist = ", ".join(f"{c} {np.mean(y == i):.3f}" for i, c in enumerate(cfg["clases"]))
    return f"Test n = {len(y):,}; clases: {dist}. Clase más probable (argmax).\n\n" + cab + "\n" + "\n".join(filas), res


def movilidad(modelos: dict, cfg: dict) -> str:
    """T2: calidad entre quienes cambian de quintil (donde la persistencia siempre falla) y entre quienes lo mantienen."""
    y = next(iter(modelos.values()))[0]["y"].astype(int)
    ids = next(iter(modelos.values()))[0]["id_trabajador"]
    q15 = _columna(cfg["tabla"], cfg["estado"], ids).astype(int)
    grupos = {"cambian de quintil": y != q15, "mantienen": y == q15}
    cab = "| Modelo | " + " | ".join(f"{g} (n={m.sum():,}): QWK / MAE" for g, m in grupos.items()) + " |"
    k = cfg["K"]
    todo = _con_base(modelos, y, cfg)
    filas = []
    for n, ps in todo.items():
        celdas = []
        for m in grupos.values():
            q = np.mean([metricas.qwk(y[m], p["P"][m].argmax(1), k) for p in ps])
            e = np.mean([metricas.mae(y[m], p["P"][m].argmax(1)) for p in ps])
            celdas.append(f"{q:.3f} / {e:.3f}")
        filas.append(f"| {n} | " + " | ".join(celdas) + " |")
    return ("QWK y MAE (media entre semillas) según el trabajador cambie o no de quintil entre 2015 y 2019.\n\n"
            + cab + "\n|---|---|---|\n" + "\n".join(filas))


def subgrupos_multi(modelos: dict, cfg: dict) -> str:
    grupos = _grupos(modelos, cfg["tabla"])
    y = next(iter(modelos.values()))[0]["y"].astype(int)
    m_ = cfg["monitor"]
    cab = "| Modelo | " + " | ".join(f"{g} (n={m.sum():,})" for g, m in grupos.items()) + " |"
    filas = [f"| {n} | " + " | ".join(
        f"{np.mean([metricas.multiclase(y[m], p['P'][m])[m_] for p in ps]):.3f}" for m in grupos.values()) + " |"
             for n, ps in modelos.items()]
    return f"{m_} por subgrupo (media entre semillas).\n\n" + cab + "\n|---|" + "---|" * len(grupos) + "\n" + "\n".join(filas)


def figura_confusion(modelos: dict, cfg: dict, salida: Path) -> None:
    """Matrices de confusión (por fila) del transformer pre-entrenado, de la mejor línea de base simple y de la persistencia."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = next(iter(modelos.values()))[0]["y"].astype(int)
    todo = _con_base(modelos, y, cfg)
    k = cfg["K"]
    elegidos = [n for n in ("Transformer pre-entrenado (L2V)", "Logística con estado actual", "Persistencia") if n in todo]
    fig, ejes = plt.subplots(1, len(elegidos), figsize=(4.6 * len(elegidos), 4.2), squeeze=False)
    for ax, n in zip(ejes[0], elegidos):
        m = metricas._confusion(y, todo[n][0]["P"].argmax(1), k)
        m = m / np.maximum(m.sum(1, keepdims=True), 1)
        ax.imshow(m, vmin=0, vmax=1, cmap="Blues")
        for i in range(k):
            for j in range(k):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", color="white" if m[i, j] > 0.55 else "black", fontsize=8)
        ax.set(title=n, xlabel="clase predicha", ylabel="clase verdadera", xticks=range(k), yticks=range(k))
        ax.set_xticklabels(cfg["clases"], fontsize=7)
        ax.set_yticklabels(cfg["clases"], fontsize=7)
    fig.tight_layout()
    fig.savefig(salida / "confusion.png", dpi=110)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tarea", default="t1", choices=list(CONFIG))
    ap.add_argument("--prefijo", help="prefijo de las corridas de una variante (p. ej. t3_uv); por defecto el de la tarea")
    ap.add_argument("--salida", help="carpeta de salida; por defecto informes/fase6, fase6b o fase8 según la tarea")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args(argv)
    cfg = dict(CONFIG[args.tarea])
    if args.prefijo:
        cfg["patrones"] = _patrones(args.prefijo)
        cfg["salida"] = INFORMES / "fase8" / args.prefijo
    runs = Path(args.runs) if Path(args.runs).is_absolute() else RAIZ / args.runs
    salida = Path(args.salida) if args.salida else cfg["salida"]
    salida.mkdir(parents=True, exist_ok=True)
    modelos = cargar_modelos(runs, cfg["patrones"])
    if not modelos:
        raise SystemExit(f"no hay predicciones en {runs}")
    resultados = {}
    if cfg["binaria"]:
        for etiqueta, archivo in cfg["etiquetas"]:
            texto, res = tabla(modelos, etiqueta, args.bootstrap, PAPER)
            (salida / archivo).write_text(texto + "\n")
            resultados[etiqueta] = res
            print(texto, "\n")
        principal = cfg["etiquetas"][0][0]
        (salida / "subgrupos.md").write_text(subgrupos(modelos, cfg["tabla"], principal) + "\n")
        (salida / "calibracion.md").write_text(calibracion(modelos, principal) + "\n")
        figuras(modelos, salida, principal)
    else:
        texto, resultados = tabla_multi(modelos, cfg, args.bootstrap)
        (salida / "tabla.md").write_text(texto + "\n")
        print(texto, "\n")
        (salida / "subgrupos.md").write_text(subgrupos_multi(modelos, cfg) + "\n")
        if cfg["estado"]:
            (salida / "movilidad.md").write_text(movilidad(modelos, cfg) + "\n")
        figura_confusion(modelos, cfg, salida)
    (salida / "resultados.json").write_text(json.dumps(resultados, indent=2, default=float))
    print(f"escrito en {salida}")


if __name__ == "__main__":
    main()
