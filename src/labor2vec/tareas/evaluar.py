"""Evaluación de T1: tabla tipo A2 del paper con IC por bootstrap, subgrupos, calibración y figuras.

    python -m labor2vec.tareas.evaluar [--runs runs] [--bootstrap 1000]

Busca las predicciones de test que escriben `ajustar` y `tabulares` (`predicciones_test.parquet`, con `p`, `y_2019`,
`y_2021`) en `runs/` según `MODELOS`, y escribe en `informes/fase6/`: `tabla_y2019.md`, `tabla_y2021.md` (la misma
predicción contra la etiqueta de 2021), `subgrupos.md`, `calibracion.md`, `resultados.json` y figuras PNG.
Los modelos con varias semillas se reportan como la media entre semillas, con el IC de esa media.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from labor2vec.tareas import metricas
from labor2vec.utiles.rutas import INFORMES, RAIZ, T1

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


def leer(carpeta: Path, split: str = "test") -> dict:
    t = pq.read_table(carpeta / f"predicciones_{split}.parquet").to_pydict()
    orden = np.argsort(t["id_trabajador"])
    return {k: np.asarray(v)[orden] for k, v in t.items()}


def cargar_modelos(runs: Path) -> dict:
    """{nombre: [predicciones por semilla]} de lo que haya en `runs`; los ids de todas las corridas deben coincidir."""
    out = {}
    for nombre, patron in MODELOS.items():
        carpetas = sorted(p for p in runs.glob(patron) if (p / "predicciones_test.parquet").exists())
        if carpetas:
            out[nombre] = [leer(c) for c in carpetas]
    ids = {n: tuple(v[0]["id_trabajador"]) for n, v in out.items()}
    assert len(set(ids.values())) <= 1, "los modelos no se evaluaron sobre los mismos trabajadores"
    return out


def formato(m: tuple, decimales: int = 3) -> str:
    return f"{m[0]:.{decimales}f} [{m[1]:.{decimales}f}, {m[2]:.{decimales}f}]"


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


def subgrupos(modelos: dict, ruta_t1=T1) -> str:
    t = pq.read_table(ruta_t1, columns=["id_trabajador", "sexo", "edad_corte"]).to_pydict()
    fila = {i: k for k, i in enumerate(t["id_trabajador"])}
    ids = next(iter(modelos.values()))[0]["id_trabajador"]
    sexo = np.array([t["sexo"][fila[int(i)]] for i in ids])
    edad = np.array([t["edad_corte"][fila[int(i)]] for i in ids])
    grupos = {"mujeres": sexo == 1, "varones": sexo == 2, "25-34": (edad >= 25) & (edad <= 34),
              "35-44": (edad >= 35) & (edad <= 44), "45-55": (edad >= 45)}
    y = next(iter(modelos.values()))[0]["y_2019"].astype(bool)
    cab = "| Modelo | " + " | ".join(f"{g} (n={m.sum():,}, tasa {y[m].mean():.3f})" for g, m in grupos.items()) + " |"
    filas = [f"| {n} | " + " | ".join(f"{np.mean([metricas.aul(y[m], p['p'][m]) for p in ps]):.3f}" for m in grupos.values()) + " |"
             for n, ps in modelos.items()]
    return "AUL por subgrupo (media entre semillas), etiqueta `y_2019`.\n\n" + cab + "\n|---|" + "---|" * len(grupos) + "\n" + "\n".join(filas)


def calibracion(modelos: dict) -> str:
    y = next(iter(modelos.values()))[0]["y_2019"].astype(bool)
    prev = y.mean()
    filas = []
    for n, ps in modelos.items():
        pc = [metricas.corregir_prevalencia(p["p"], prev) for p in ps]
        filas.append(f"| {n} | {np.mean([metricas.brier(y, p) for p in pc]):.4f} |")
    base = metricas.brier(y, np.full(len(y), prev))
    return (f"Brier con las probabilidades llevadas a la prevalencia real ({prev:.3f}); constante = {base:.4f}.\n\n"
            "| Modelo | Brier |\n|---|---|\n" + "\n".join(filas))


def figuras(modelos: dict, salida: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = next(iter(modelos.values()))[0]["y_2019"].astype(bool)
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
    ejes[0].set(title="Ganancia acumulada (AUL)", xlabel="fracción de trabajadores con mayor riesgo", ylabel="fracción de salidas")
    ejes[1].plot([0, 1], [0, 1], "k:")
    ejes[1].set(title="Curva ROC", xlabel="falsos positivos", ylabel="verdaderos positivos")
    ejes[2].plot([0, 0.6], [0, 0.6], "k:")
    ejes[2].set(title="Calibración (prevalencia corregida)", xlabel="probabilidad predicha", ylabel="frecuencia observada")
    ejes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(salida / "curvas.png", dpi=110)


def _roc(y, p):
    orden = np.argsort(-p, kind="mergesort")
    ys = y[orden]
    return np.concatenate([[0], np.cumsum(~ys) / (~ys).sum()]), np.concatenate([[0], np.cumsum(ys) / ys.sum()])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args(argv)
    runs = Path(args.runs) if Path(args.runs).is_absolute() else RAIZ / args.runs
    salida = INFORMES / "fase6"
    salida.mkdir(parents=True, exist_ok=True)
    modelos = cargar_modelos(runs)
    if not modelos:
        raise SystemExit(f"no hay predicciones en {runs}")
    resultados = {}
    for etiqueta, archivo in (("y_2019", "tabla_y2019.md"), ("y_2021", "tabla_y2021.md")):
        texto, res = tabla(modelos, etiqueta, args.bootstrap, PAPER)
        (salida / archivo).write_text(texto + "\n")
        resultados[etiqueta] = res
        print(texto, "\n")
    (salida / "subgrupos.md").write_text(subgrupos(modelos) + "\n")
    (salida / "calibracion.md").write_text(calibracion(modelos) + "\n")
    (salida / "resultados.json").write_text(json.dumps(resultados, indent=2, default=float))
    figuras(modelos, salida)
    print(f"escrito en {salida}")


if __name__ == "__main__":
    main()
