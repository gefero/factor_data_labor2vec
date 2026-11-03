"""Métricas sobre la matriz de embeddings de conceptos: orden, pureza de vecindades, robustez entre modelos, hubness.

Todas trabajan con distancia coseno y tomando sólo los tokens que se les pasan (nunca los especiales). Las que tienen
un nulo devuelven (observado, media del nulo, p-valor empírico de una cola): p = (1 + #{nulo >= obs}) / (1 + n).
"""

import numpy as np
from scipy import stats


def normalizar(e: np.ndarray) -> np.ndarray:
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def distancias_coseno(e: np.ndarray) -> np.ndarray:
    x = normalizar(e)
    return np.clip(1.0 - x @ x.T, 0.0, 2.0)


def vecinos(e: np.ndarray, k: int) -> np.ndarray:
    """Índices (n, k) de los k vecinos más cercanos por coseno de cada fila, sin contarse a sí misma."""
    x = normalizar(e)
    s = x @ x.T
    np.fill_diagonal(s, -np.inf)
    return np.argsort(-s, axis=1)[:, :k]


def p_empirico(obs: float, nulo: np.ndarray) -> float:
    return float((1 + np.sum(nulo >= obs)) / (1 + len(nulo)))


# ---- orden ----------------------------------------------------------------------------------------------------
def orden_pc1(e: np.ndarray, valores: np.ndarray, n_perm: int = 1000, rng=None) -> dict:
    """|Spearman| entre el valor ordinal y la primera componente principal, con nulo de valores permutados."""
    rng = rng or np.random.default_rng(0)
    x = e - e.mean(0)
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    pc1 = x @ vt[0]
    obs = abs(stats.spearmanr(pc1, valores).statistic)
    nulo = np.array([abs(stats.spearmanr(pc1, rng.permutation(valores)).statistic) for _ in range(n_perm)])
    return {"rho": obs, "nulo": float(nulo.mean()), "p": p_empirico(obs, nulo), "var_pc1": float(s[0] ** 2 / (s**2).sum())}


def consistencia_ordinal(e: np.ndarray, valores: np.ndarray, k: int = 4, ventana: float = 5, n_perm: int = 1000,
                         rng=None) -> dict:
    """Proporción de los k vecinos más cercanos cuyo valor difiere en <= `ventana`, con nulo de valores permutados."""
    rng = rng or np.random.default_rng(0)
    nb = vecinos(e, k)

    def medida(v):
        return float(np.mean(np.abs(v[nb] - v[:, None]) <= ventana))

    obs = medida(valores)
    nulo = np.array([medida(rng.permutation(valores)) for _ in range(n_perm)])
    return {"prop": obs, "nulo": float(nulo.mean()), "p": p_empirico(obs, nulo)}


def orden_pares(e: np.ndarray) -> dict:
    """Para pocos niveles ordenados (por orden de fila): Spearman entre la distancia coseno de cada par y |i - j|."""
    d = distancias_coseno(e)
    iu = np.triu_indices(len(e), 1)
    ordinal = np.abs(iu[0] - iu[1])
    return {"rho": float(stats.spearmanr(d[iu], ordinal).statistic), "n_pares": len(ordinal)}


def orden_distancias(e: np.ndarray, valores: np.ndarray, n_perm: int = 200, rng=None) -> dict:
    """Spearman entre la distancia coseno de cada par de tokens y la diferencia absoluta de sus valores ordinales.

    A diferencia de la primera componente principal, no supone que el orden sea una recta: sirve para curvas que se
    doblan. Nulo: valores permutados."""
    rng = rng or np.random.default_rng(0)
    d = triangulo(distancias_coseno(e))
    iu = np.triu_indices(len(e), 1)
    valores = np.asarray(valores, dtype=float)
    obs = float(stats.spearmanr(d, np.abs(valores[iu[0]] - valores[iu[1]])).statistic)
    nulo = np.array([stats.spearmanr(d, np.abs((v := rng.permutation(valores))[iu[0]] - v[iu[1]])).statistic
                     for _ in range(n_perm)])
    return {"rho": obs, "nulo": float(nulo.mean()), "p": p_empirico(obs, nulo)}


# ---- pureza de vecindades ---------------------------------------------------------------------------------------
def pureza(e: np.ndarray, etiquetas: np.ndarray, k: int, n_perm: int = 1000, rng=None) -> dict:
    """Proporción de los k vecinos más cercanos (dentro de `e`) con la misma etiqueta, con nulo de etiquetas permutadas."""
    rng = rng or np.random.default_rng(0)
    nb = vecinos(e, k)
    etiquetas = np.asarray(etiquetas)

    def medida(et):
        return float(np.mean(et[nb] == et[:, None]))

    obs = medida(etiquetas)
    nulo = np.array([medida(rng.permutation(etiquetas)) for _ in range(n_perm)])
    return {"pureza": obs, "nulo": float(nulo.mean()), "p": p_empirico(obs, nulo)}


# ---- robustez entre modelos (protocolo de emb_robust.ipynb de life2vec) ----------------------------------------------
def triangulo(d: np.ndarray) -> np.ndarray:
    return d[np.triu_indices(d.shape[0], 1)]


def rho_modelos(d1: np.ndarray, d2: np.ndarray) -> float:
    return float(stats.spearmanr(triangulo(d1), triangulo(d2)).statistic)


def nulo_columnas_permutadas(e: np.ndarray, rng) -> np.ndarray:
    """Cada columna (dimensión) se permuta de forma independiente: conserva las distribuciones y rompe la estructura."""
    p = e.copy()
    for j in range(p.shape[1]):
        p[:, j] = rng.permutation(p[:, j])
    return p


def nulo_gaussiano(e: np.ndarray, rng) -> np.ndarray:
    p = nulo_columnas_permutadas(e, rng)
    return rng.normal(p.mean(), p.std(), size=p.shape)


def test_permutacion(d1: np.ndarray, d2: np.ndarray, n_iter: int = 5000, rng=None) -> dict:
    """Test de permutación Monte Carlo de una cola: se permutan filas y columnas de `d1` (mismos índices) y se compara
    el Spearman con el observado. Equivale al `permutation_test` del notebook original, con los rangos precalculados."""
    rng = rng or np.random.default_rng(0)
    n = d1.shape[0]
    iu = np.triu_indices(n, 1)
    r1 = np.zeros((n, n))
    r1[iu] = stats.rankdata(d1[iu])
    r1 = r1 + r1.T
    r2 = stats.rankdata(d2[iu])
    r2 = (r2 - r2.mean()) / np.linalg.norm(r2 - r2.mean())

    def pearson(v):
        v = v - v.mean()
        return float(v @ r2 / np.linalg.norm(v))

    obs = pearson(r1[iu])
    nulo = np.array([pearson(r1[np.ix_(p, p)][iu]) for p in (rng.permutation(n) for _ in range(n_iter))])
    return {"rho": obs, "nulo_medio": float(nulo.mean()), "nulo_max": float(nulo.max()), "p": p_empirico(obs, nulo)}


def fdr_bh(p: np.ndarray) -> np.ndarray:
    """P-valores ajustados por Benjamini-Hochberg."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    orden = np.argsort(p)
    aj = p[orden] * m / np.arange(1, m + 1)
    aj = np.minimum.accumulate(aj[::-1])[::-1]
    out = np.empty(m)
    out[orden] = np.minimum(aj, 1.0)
    return out


# ---- hubness (emb_hubs.ipynb) --------------------------------------------------------------------------------------
def grado_entrada(e: np.ndarray, k: int = 5) -> np.ndarray:
    """Cuántas veces aparece cada token entre los k vecinos más cercanos de otro."""
    return np.bincount(vecinos(e, k).ravel(), minlength=len(e))


def resumen_hubness(grados: np.ndarray, grupos: dict[str, np.ndarray]) -> dict:
    """Asimetría global y, por grupo de tokens, mediana, máximo y proporción con grado de entrada 0."""
    out = {"asimetria": float(stats.skew(grados)), "maximo": int(grados.max()),
           "prop_cero": float(np.mean(grados == 0)), "por_grupo": {}}
    for nombre, idx in grupos.items():
        g = grados[idx]
        out["por_grupo"][nombre] = {"n": len(idx), "mediana": float(np.median(g)), "max": int(g.max()),
                                    "prop_cero": float(np.mean(g == 0))}
    return out
