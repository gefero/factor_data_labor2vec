"""Comparaciones de las ablaciones de la fase 8 (plans/PLAN_fase_8.md), con la diferencia pareada y su IC por bootstrap.

    python -m labor2vec.tareas.ablaciones [--runs runs] [--bootstrap 1000] [--salida informes/fase8]

Lee las predicciones de test que ya escribieron `ajustar`, `tabulares` y `predecir` y escribe `comparaciones.md` y
`comparaciones.json`. Las secciones sin corridas se omiten, así que se puede correr con resultados parciales.
1. Control temporal sin reajustar (`runs/ctrl_{ninguno,tiempo,orden}/…`): el mismo modelo con la entrada alterada.
2. Reajuste con el tiempo destruido (`<tarea>_tiempo_l2v_s*`) contra el modelo original, en T3 y T4.
3. Sin atributos de último valor (`<tarea>_uv_…`) contra el modelo con todos los atributos, para cada familia de modelos.
4. Lo mismo, separado por si la relación principal de 2015 continúa después de 2015-12 (`continuidad.parquet`).
La métrica de cada tarea es la de monitoreo: AUL (T1, T3, T4c), QWK (T2) o AUC macro (T4, T2c, T3d).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from labor2vec.tareas import evaluar, metricas
from labor2vec.utiles.rutas import INFORMES, RAIZ
from labor2vec.utiles.rutas import TAREAS as DIR_TAREAS

TAREAS = ("t1", "t2", "t3", "t4")
METRICA = {"t1": "aul", "t2": "qwk", "t3": "aul", "t4": "auc_macro"}
NOMBRE_METRICA = {"aul": "AUL", "qwk": "QWK", "auc_macro": "AUC macro"}
ETIQUETA = {"t1": "y_2019"}  # en las demás tareas la etiqueta es `y`
FAMILIAS = {"Transformer pre-entrenado": "{l2v}", "Transformer sin pre-entrenar": "{t}_scratch_s*", "GRU": "{t}_gru_s*",
            "FFNN": "{t}_tab/ffnn_s*", "Regresión logística": "{t}_tab/logistica"}


def patron(familia: str, tarea: str, prefijo: str | None = None) -> str:
    """Patrón de carpetas de una familia de modelos. `prefijo` (p. ej. `t3_uv`) reemplaza al de la tarea; T1 usa las
    corridas con lr 0,001 (`t1_l2v_lr0001*`), las demás `<tarea>_l2v_s*`."""
    t = prefijo or tarea
    l2v = "t1_l2v_lr0001*" if (tarea == "t1" and not prefijo) else f"{t}_l2v_s*"
    return FAMILIAS[familia].format(t=t, l2v=l2v)


def funcion_metrica(tarea: str):
    """f(y, p) -> número, con `p` el vector de probabilidades (binaria) o la matriz (n, K) (multiclase)."""
    m = METRICA[tarea]
    if m == "aul":
        return metricas.aul
    return lambda y, P: metricas.multiclase(y, P)[m]


def cargar(runs: Path, pat: str, tarea: str) -> dict | None:
    """{ids, y, ps (lista de predicciones por semilla)} de las carpetas que cumplen `pat`, o None si no hay."""
    carpetas = sorted(p for p in Path(runs).glob(pat) if (p / "predicciones_test.parquet").exists())
    if not carpetas:
        return None
    preds = [evaluar.leer(c) for c in carpetas]
    clave = "p" if "p" in preds[0] else "P"
    return {"ids": preds[0]["id_trabajador"], "y": preds[0][ETIQUETA.get(tarea, "y")].astype(int),
            "ps": [p[clave] for p in preds], "n": len(preds)}


def _media(f, a: dict) -> float:
    return float(np.mean([f(a["y"], p) for p in a["ps"]]))


def comparar(f, a: dict, b: dict, n_boot: int) -> dict:
    """A menos B en la métrica `f`, sobre las mismas personas."""
    assert np.array_equal(a["ids"], b["ids"]), "las predicciones no son de las mismas personas"
    d, lo, hi = metricas.bootstrap_diferencia(a["y"], a["ps"], b["ps"], f, n_boot, semilla=1)
    return {"a": _media(f, a), "b": _media(f, b), "dif": d, "lo": lo, "hi": hi}


def _celda(r: dict) -> str:
    marca = "" if r["lo"] <= 0 <= r["hi"] else " *"
    return f"{r['a']:.3f} | {r['b']:.3f} | {r['dif']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]{marca}"


def _tabla(titulo: str, cab_a: str, cab_b: str, filas: list[tuple[str, dict]], nota: str = "") -> str:
    if not filas:
        return ""
    t = [f"{titulo}\n", nota + "\n" if nota else "", f"| | {cab_a} | {cab_b} | diferencia [IC 95 %] |", "|---|---|---|---|"]
    t += [f"| {n} | {_celda(r)} |" for n, r in filas]
    return "\n".join(x for x in t if x != "") + "\n"


# ---- secciones ------------------------------------------------------------------------------------------------------
def control_temporal(runs: Path, n_boot: int, res: dict) -> str:
    """El mismo modelo ajustado, con la entrada normal y con el control (sin reajustar)."""
    partes = []
    for control in ("tiempo", "orden"):
        filas = []
        for t in TAREAS:
            pat = patron("Transformer pre-entrenado", t)
            a, b = cargar(runs / "ctrl_ninguno", pat, t), cargar(runs / f"ctrl_{control}", pat, t)
            if a and b:
                filas.append((f"{t.upper()} ({NOMBRE_METRICA[METRICA[t]]})", comparar(funcion_metrica(t), a, b, n_boot)))
                res.setdefault("control_temporal", {})[f"{t}_{control}"] = filas[-1][1]
        partes.append(_tabla(f"### Control «{control}», sin reajustar", "entrada normal", f"control {control}", filas))
    return "\n".join(p for p in partes if p)


def reajuste_tiempo(runs: Path, n_boot: int, res: dict) -> str:
    """Reajustado desde cero con el tiempo destruido, contra el modelo original."""
    filas = []
    for t in ("t3", "t4"):
        a = cargar(runs, patron("Transformer pre-entrenado", t), t)
        b = cargar(runs, f"{t}_tiempo_l2v_s*", t)
        if a and b:
            filas.append((f"{t.upper()} ({NOMBRE_METRICA[METRICA[t]]})", comparar(funcion_metrica(t), a, b, n_boot)))
            res.setdefault("reajuste_tiempo", {})[t] = filas[-1][1]
    return _tabla("### Reajuste con el tiempo destruido", "original", "tiempo destruido", filas,
                  "Se reajustó el transformer pre-entrenado con las fechas y edades permutadas entre los eventos.")


def sin_ultimo_valor(runs: Path, n_boot: int, res: dict) -> str:
    partes = []
    for t in TAREAS:
        filas = []
        for fam in FAMILIAS:
            a, b = cargar(runs, patron(fam, t), t), cargar(runs, patron(fam, t, f"{t}_uv"), t)
            if a and b:
                filas.append((fam, comparar(funcion_metrica(t), a, b, n_boot)))
                res.setdefault("sin_ultimo_valor", {})[f"{t}_{fam}"] = filas[-1][1]
        partes.append(_tabla(f"### {t.upper()}: con y sin atributos de último valor ({NOMBRE_METRICA[METRICA[t]]})",
                             "con todos", "sin último valor", filas))
    return "\n".join(p for p in partes if p)


def continuidad(runs: Path, n_boot: int, res: dict, ruta=DIR_TAREAS / "continuidad.parquet") -> str:
    """Con y sin atributos de último valor, por separado entre quienes tienen la relación principal vigente después del
    corte (la fuga es posible) y quienes la terminaron antes (no lo es)."""
    if not Path(ruta).exists():
        return ""
    tabla = pq.read_table(ruta).to_pydict()
    fila = {i: k for k, i in enumerate(tabla["id_trabajador"])}
    partes = []
    for t in TAREAS:
        for fam in ("Transformer pre-entrenado", "FFNN"):
            a, b = cargar(runs, patron(fam, t), t), cargar(runs, patron(fam, t, f"{t}_uv"), t)
            if not (a and b):
                continue
            continua = np.array([bool(tabla["continua"][fila[int(i)]]) if int(i) in fila else False for i in a["ids"]])
            filas = []
            for nombre, m in (("continúa después del corte", continua), ("terminó en 2015-12 o antes", ~continua)):
                sub = lambda x, m=m: {**x, "ids": x["ids"][m], "y": x["y"][m], "ps": [p[m] for p in x["ps"]]}  # noqa: E731
                if len(np.unique(a["y"][m])) < 2:
                    continue
                filas.append((f"{nombre} (n = {int(m.sum()):,})", comparar(funcion_metrica(t), sub(a), sub(b), n_boot)))
                res.setdefault("continuidad", {})[f"{t}_{fam}_{'continua' if nombre[0] == 'c' else 'termino'}"] = filas[-1][1]
            partes.append(_tabla(f"### {t.upper()}, {fam}: por continuidad de la relación principal "
                                 f"({NOMBRE_METRICA[METRICA[t]]})", "con todos", "sin último valor", filas))
    return "\n".join(p for p in partes if p)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--salida", default=str(INFORMES / "fase8"))
    args = ap.parse_args(argv)
    runs = Path(args.runs) if Path(args.runs).is_absolute() else RAIZ / args.runs
    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    res: dict = {}
    secciones = [control_temporal(runs, args.bootstrap, res), reajuste_tiempo(runs, args.bootstrap, res),
                 sin_ultimo_valor(runs, args.bootstrap, res), continuidad(runs, args.bootstrap, res)]
    texto = ("# Comparaciones de la fase 8 (generado por `tareas.ablaciones`)\n\n"
             "Media entre semillas en test; diferencia pareada con IC al 95 % por bootstrap sobre las mismas personas. "
             "`*` = el IC no incluye el 0.\n\n" + "\n".join(s for s in secciones if s))
    (salida / "comparaciones.md").write_text(texto)
    (salida / "comparaciones.json").write_text(json.dumps(res, indent=2, default=float))
    print(texto)


if __name__ == "__main__":
    main()
