"""TCAV (Kim et al., 2018) sobre el espacio de personas, con el protocolo del life2vec original.

Un concepto es un grupo de personas (p. ej. «mujeres»). Su vector de concepto (CAV) es la dirección, en el estado promedio
de la última capa del encoder, que separa al grupo de personas al azar (regresión logística). El puntaje TCAV de un
modelo es la fracción de personas para las que mover el estado de todos sus tokens en esa dirección sube el logit, es decir
`g · v > 0` con `g` la suma sobre tokens del gradiente del logit respecto de la capa (se calcula en `extraer`).
"""

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression


def cavs(x: np.ndarray, concepto: np.ndarray, n: int = 50, tam: int = 500, rng=None) -> tuple[np.ndarray, np.ndarray]:
    """`n` CAVs de longitud 1 (n, d) y su exactitud en datos retenidos (n,).

    `x`: estados promedio (personas, d); `concepto`: máscara booleana de las personas del concepto. Cada CAV usa `tam`
    personas del concepto contra `tam` al azar del resto; 80 % para ajustar y 20 % para medir la exactitud. Con
    `concepto=None` es el nulo: dos grupos al azar sin relación con ningún concepto."""
    rng = rng or np.random.default_rng(0)
    idx = np.arange(len(x))
    vs, accs = [], []
    for _ in range(n):
        if concepto is None:
            ele = rng.choice(idx, 2 * tam, replace=False)
            pos, neg = ele[:tam], ele[tam:]
        else:
            pos = rng.choice(idx[concepto], min(tam, int(concepto.sum())), replace=False)
            neg = rng.choice(idx[~concepto], len(pos), replace=False)
        X = np.concatenate([x[pos], x[neg]])
        y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        orden = rng.permutation(len(y))
        corte = int(0.8 * len(y))
        tr, te = orden[:corte], orden[corte:]
        clf = LogisticRegression(max_iter=1000).fit(X[tr], y[tr])
        v = clf.coef_[0]
        vs.append(v / np.linalg.norm(v))
        accs.append(clf.score(X[te], y[te]))
    return np.array(vs), np.array(accs)


def puntajes(g: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Fracción de personas con derivada positiva, una por CAV: g (personas, d), v (n, d) → (n,)."""
    return (g @ v.T > 0).mean(0)


def comparar(p_concepto: np.ndarray, p_nulo: np.ndarray, n_pruebas: int = 1) -> dict:
    """Puntajes de un concepto contra los del nulo: media, t de Welch y p con Bonferroni por `n_pruebas`."""
    t = stats.ttest_ind(p_concepto, p_nulo, equal_var=False)
    return {"tcav": float(p_concepto.mean()), "sd": float(p_concepto.std(ddof=1)), "nulo": float(p_nulo.mean()),
            "t": float(t.statistic), "p": float(min(1.0, t.pvalue * n_pruebas))}


def tabla(x_val: np.ndarray, g: np.ndarray, y: np.ndarray, conceptos_val: dict, n: int = 50, tam: int = 500,
          semilla: int = 0) -> list[dict]:
    """Un renglón por concepto (y grupo de `y`): puntaje TCAV, exactitud del CAV y comparación con el nulo.

    `x_val`: estados promedio de val; `conceptos_val`: {nombre: máscara sobre val}; `g`/`y`: gradientes y etiquetas del
    test. Los CAVs se ajustan en val y se aplican a los gradientes del test, separados por clase real."""
    rng = np.random.default_rng(semilla)
    v0, _ = cavs(x_val, None, n, tam, rng)
    nulo = {c: puntajes(g[y == c], v0) for c in np.unique(y)}
    filas = []
    for nombre, mascara in conceptos_val.items():
        if mascara.sum() < 50:
            continue
        v, acc = cavs(x_val, mascara, n, tam, rng)
        for c in np.unique(y):
            filas.append({"concepto": nombre, "clase": int(c), "exactitud": float(acc.mean()),
                          **comparar(puntajes(g[y == c], v), nulo[c], len(conceptos_val))})
    return filas
