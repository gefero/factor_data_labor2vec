"""Métricas de clasificación binaria para T1 (plans/PLAN_fase_6.md, sección 6.4). Sólo numpy: se usan también durante el
entrenamiento, así que no importa polars ni scipy.

- AUL (área bajo la curva de lift): promedio, sobre los positivos, de la fracción de todas las observaciones con puntaje
  menor (los empates cuentan 1/2). Equivale a c/2 + (1 - c) AUC, con c la proporción de positivos. No necesita umbral.
- Medidas corregidas de Jain, White y Radivojac (2017), como en `src/transformer/metrics.py` del life2vec original:
  `alfa` es la proporción de positivos entre los no etiquetados y `beta` la de positivos entre los etiquetados como
  positivos. Con etiquetas observadas (`alfa=0`, `beta=1`) el C-MCC es el MCC común.
"""

import numpy as np


def _rangos(x: np.ndarray) -> np.ndarray:
    """Rangos promedio (1 = el menor), con los empates repartidos."""
    orden = np.argsort(x, kind="mergesort")
    xs = x[orden]
    nuevo = np.concatenate([[True], xs[1:] != xs[:-1]])
    inicio = np.flatnonzero(nuevo)
    fin = np.concatenate([inicio[1:], [len(x)]])
    promedio = (inicio + 1 + fin) / 2  # rango promedio de cada grupo de empatados
    r = np.empty(len(x))
    r[orden] = np.repeat(promedio, fin - inicio)
    return r


def _suma_rangos_pos(y: np.ndarray, s: np.ndarray) -> tuple[float, int, int]:
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=np.float64)
    n_pos = int(y.sum())
    return _rangos(s)[y].sum(), n_pos, len(y) - n_pos


def auc(y, s) -> float:
    suma, n_pos, n_neg = _suma_rangos_pos(y, s)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (suma - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def aul(y, s) -> float:
    suma, n_pos, n_neg = _suma_rangos_pos(y, s)
    if n_pos == 0:
        return float("nan")
    return (suma - 0.5 * n_pos) / (n_pos * (n_pos + n_neg))


def pr_auc(y, s) -> float:
    """Precisión promedio (área bajo la curva precisión-exhaustividad, escalonada)."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=np.float64)
    if y.sum() == 0:
        return float("nan")
    orden = np.argsort(-s, kind="mergesort")
    ys, ss = y[orden], s[orden]
    fin_grupo = np.concatenate([ss[1:] != ss[:-1], [True]])  # los empates se cortan juntos
    tp, n = np.cumsum(ys)[fin_grupo], (np.arange(len(ys)) + 1)[fin_grupo]
    precision, exhaustividad = tp / n, tp / y.sum()
    return float(np.sum(np.diff(np.concatenate([[0], exhaustividad])) * precision))


def corregidas(y, p, umbral: float = 0.5, alfa: float = 0.0, beta: float = 1.0) -> dict:
    """C-MCC, F1, exactitud balanceada corregidas y MCC común, con el umbral sobre p(positivo)."""
    y = np.asarray(y).astype(bool)
    pred = np.asarray(p) >= umbral
    tp, fn = int((pred & y).sum()), int((~pred & y).sum())
    fp, tn = int((pred & ~y).sum()), int((~pred & ~y).sum())
    n = tp + fn + fp + tn
    gamma = tp / (tp + fn) if tp + fn else 0.0   # tasa de verdaderos positivos
    nu = fp / (fp + tn) if fp + tn else 0.0      # tasa de falsos positivos
    c, theta = (tp + fn) / n, (tp + fp) / n
    gamma_cr = ((1 - alfa) * gamma - (1 - beta) * nu) / (beta - alfa)
    nu_cr = (beta * nu - alfa * gamma) / (beta - alfa)
    c_cr = beta * c + (1 - c) * alfa
    den = theta * (1 - theta)
    cmcc = np.sqrt(c_cr * (1 - c_cr) / den) * (gamma_cr - nu_cr) if den > 0 else 0.0
    f1 = 2 * c_cr * gamma_cr / (c_cr + theta) if c_cr + theta > 0 else 0.0
    mcc_comun = np.sqrt(c * (1 - c) / den) * (gamma - nu) if den > 0 and 0 < c < 1 else 0.0
    return {"c_mcc": float(cmcc), "f1_corr": float(f1), "exactitud_corr": float((1 + gamma_cr - nu_cr) / 2),
            "mcc": float(mcc_comun)}


def umbral_optimo(y, p, n_umbrales: int = 99) -> float:
    """Umbral que maximiza el MCC (se elige en validación y se aplica a test)."""
    candidatos = np.unique(np.quantile(p, np.linspace(0.01, 0.99, n_umbrales)))
    return float(max(candidatos, key=lambda u: corregidas(y, p, u)["mcc"]))


def corregir_prevalencia(p, prevalencia: float, prevalencia_entrenamiento: float = 0.5):
    """Lleva probabilidades de un modelo entrenado con otra prevalencia (50/50 por el muestreo balanceado) a la real."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-9, 1 - 1e-9)
    odds = p / (1 - p) * (prevalencia / (1 - prevalencia)) / (prevalencia_entrenamiento / (1 - prevalencia_entrenamiento))
    return odds / (1 + odds)


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y).astype(float)) ** 2))


def calibracion(y, p, n_cajas: int = 10) -> np.ndarray:
    """Filas (p media, frecuencia observada, n) por cajas de igual cantidad de observaciones."""
    y, p = np.asarray(y).astype(float), np.asarray(p, dtype=np.float64)
    orden = np.argsort(p, kind="mergesort")
    return np.array([(p[g].mean(), y[g].mean(), len(g)) for g in np.array_split(orden, n_cajas) if len(g)])


def todas(y, p, umbral: float = 0.5) -> dict:
    return {"aul": aul(y, p), "auc": auc(y, p), "pr_auc": pr_auc(y, p), **corregidas(y, p, umbral)}


def bootstrap(y, ps: list, n: int = 1000, semilla: int = 0, umbral: float = 0.5) -> dict:
    """IC al 95 % por bootstrap estratificado (positivos y negativos se remuestrean por separado).

    `ps` es una lista de puntajes (uno por semilla del mismo modelo): en cada réplica se calcula cada métrica para cada
    semilla y se promedia, de modo que el intervalo corresponde a la media entre semillas. Devuelve
    {métrica: (media entre semillas en la muestra completa, inferior, superior)}.
    """
    y = np.asarray(y).astype(bool)
    ps = [np.asarray(p, dtype=np.float64) for p in ps]
    rng = np.random.default_rng(semilla)
    pos, neg = np.flatnonzero(y), np.flatnonzero(~y)
    base = {k: np.mean([todas(y, p, umbral)[k] for p in ps]) for k in todas(y, ps[0], umbral)}
    muestras = {k: [] for k in base}
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        res = [todas(y[idx], p[idx], umbral) for p in ps]
        for k in base:
            muestras[k].append(np.mean([r[k] for r in res]))
    return {k: (float(base[k]), *np.percentile(muestras[k], [2.5, 97.5]).tolist()) for k in base}
