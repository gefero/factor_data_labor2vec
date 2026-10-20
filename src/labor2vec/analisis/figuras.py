"""Figuras del espacio de conceptos: mapa PaCMAP, dispersión entre modelos e histograma de hubness."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pacmap  # noqa: E402

from labor2vec.analisis.metricas import distancias_coseno  # noqa: E402

NOMBRES_CATEGORIA = {"sexo": "Sexo", "nacimiento": "Año de nacimiento", "salario": "Salario (percentil)",
                     "tamaño": "Tamaño de empresa", "rama": "Rama (r34)", "rama2": "Rama (r32)",
                     "provincia": "Provincia", "antigüedad": "Antigüedad de la empresa"}


def proyectar(e: np.ndarray) -> np.ndarray:
    """PaCMAP 2D con los parámetros del notebook concept_space.ipynb de life2vec."""
    return pacmap.PaCMAP(n_components=2, n_neighbors=None, random_state=0, MN_ratio=1, FP_ratio=10,
                         distance="angular", lr=0.5).fit_transform(e.astype("float32"))


def _guardar(fig, ruta):
    fig.tight_layout()
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


def mapa_categorias(xy, tabla, ruta, titulo: str) -> None:
    cats = list(NOMBRES_CATEGORIA)
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, c in enumerate(cats):
        m = tabla["categoria"].to_numpy() == c
        if m.any():
            ax.scatter(xy[m, 0], xy[m, 1], s=22, color=cmap(i), label=f"{NOMBRES_CATEGORIA[c]} ({m.sum()})",
                       edgecolors="white", linewidths=0.4, alpha=0.9)
    ax.set_title(titulo)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=8, loc="best", frameon=False)
    _guardar(fig, ruta)


def paneles(xy, tabla, ruta, titulo: str) -> None:
    cat = tabla["categoria"].to_numpy()
    fig, axs = plt.subplots(2, 2, figsize=(11, 9.5))
    for ax in axs.ravel():
        ax.scatter(xy[:, 0], xy[:, 1], s=6, color="#cccccc")
        ax.set_xticks([])
        ax.set_yticks([])
    for ax, c, nombre in ((axs[0, 0], "salario", "Salario: percentil"), (axs[0, 1], "nacimiento", "Año de nacimiento")):
        m = cat == c
        sc = ax.scatter(xy[m, 0], xy[m, 1], s=30, c=tabla["valor"].to_numpy()[m].astype(float), cmap="viridis",
                        edgecolors="white", linewidths=0.4)
        fig.colorbar(sc, ax=ax, shrink=0.8)
        ax.set_title(nombre)
    ax = axs[1, 0]
    m = cat == "provincia"
    regiones = sorted(set(tabla["region"].to_numpy()[m]))
    cmap = plt.get_cmap("tab10")
    for i, r in enumerate(regiones):
        mm = m & (tabla["region"].to_numpy() == r)
        ax.scatter(xy[mm, 0], xy[mm, 1], s=50, color=cmap(i), label=r, edgecolors="white", linewidths=0.5)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Provincia: región")
    ax = axs[1, 1]
    m = cat == "rama"
    letras = tabla["letra"].to_numpy()
    sc = ax.scatter(xy[m, 0], xy[m, 1], s=18, c=letras[m].astype(float), cmap="tab20", edgecolors="white",
                    linewidths=0.3)
    fig.colorbar(sc, ax=ax, shrink=0.8, label="letra (sector)")
    ax.set_title("Rama (r34): letra del sector")
    fig.suptitle(titulo)
    _guardar(fig, ruta)


def dispersion_modelos(distancias: dict, pares: list[tuple[str, str]], ruta, n_pares: int = 10000, rng=None) -> None:
    """Distancia coseno de pares de tokens al azar en un modelo contra la del mismo par en otro, como en el original."""
    rng = rng or np.random.default_rng(0)
    n = next(iter(distancias.values())).shape[0]
    i, j = np.triu_indices(n, 1)
    sel = rng.choice(len(i), size=min(n_pares, len(i)), replace=False)
    fig, axs = plt.subplots(1, len(pares), figsize=(4.2 * len(pares), 4.2), squeeze=False)
    for ax, (a, b) in zip(axs[0], pares):
        ax.scatter(distancias[a][i[sel], j[sel]], distancias[b][i[sel], j[sel]], s=2, alpha=0.2, color="#3b6ea5")
        ax.set_xlabel(f"distancia coseno ({a})")
        ax.set_ylabel(f"distancia coseno ({b})")
        ax.set_title(f"{a} contra {b}")
    _guardar(fig, ruta)


def hubness(grados: dict, ruta, k: int) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    maxg = max(int(g.max()) for g in grados.values())
    bins = np.arange(0, maxg + 2) - 0.5
    for nombre, g in grados.items():
        ax.hist(g, bins=bins, alpha=0.5, label=f"{nombre} (mediana {np.median(g):.0f})")
    ax.axvline(k, color="k", linestyle="--", linewidth=0.8)
    ax.set_xlabel(f"grado de entrada en el grafo de {k} vecinos")
    ax.set_ylabel("tokens")
    ax.set_title("Hubness: la línea punteada marca el grado medio")
    ax.legend(fontsize=8, frameon=False)
    _guardar(fig, ruta)


def distancias(e: np.ndarray) -> np.ndarray:
    return distancias_coseno(e)
