"""Evaluación del espacio de conceptos (fase 5): ordenamiento, pureza de vecindades, robustez entre semillas, hubness.

    python -m labor2vec.analisis.conceptos [--n-perm 1000] [--n-iter 5000]

Usa los `modelo_final.pt` de runs/*_vista/ (copias locales de las corridas del cluster) y escribe las tablas en
`informes/fase5/resultados.md` y las figuras en `informes/fase5/`.
"""

import argparse
import time

import numpy as np

from labor2vec.analisis import figuras
from labor2vec.analisis import metricas as m
from labor2vec.analisis.embeddings import cargar_conceptos, matriz_embeddings
from labor2vec.tokens.vocabulario import N_ESPECIALES
from labor2vec.utiles.rutas import INFORMES, RAIZ

MODELOS = {
    "s0": RAIZ / "runs/pre_150ep_vista/modelo_final.pt",
    "s1": RAIZ / "runs/pre_150ep_s1_vista/modelo_final.pt",
    "s2": RAIZ / "runs/pre_150ep_s2_vista/modelo_final.pt",
    "ep30": RAIZ / "runs/pre_30ep_vista/modelo_final.pt",
}
SEMILLAS = ("s0", "s1", "s2")
K_RAMA, K_PROV, K_HUB = 5, 3, 5


def _fmt_p(p: float) -> str:
    return f"{p:.4f}" if p >= 0.0001 else "< 0,0001"


def metricas_modelo(e: np.ndarray, c, n_perm: int, rng) -> dict:
    """Todas las medidas de un modelo. `e` ya excluye los tokens especiales (fila i = token i + N_ESPECIALES)."""
    t = c.tabla
    cat = t["categoria"].to_numpy()
    valor = t["valor"].to_numpy()
    out = {}
    # salarios: del percentil 1 al 100 en una curva; REM_000 (remuneración cero) aparte
    ms = (cat == "salario") & (valor >= 1)
    out["salario_pc1"] = m.orden_pc1(e[ms], valor[ms].astype(float), n_perm, rng)
    out["salario_vec"] = m.consistencia_ordinal(e[ms], valor[ms].astype(float), k=4, ventana=5, n_perm=n_perm, rng=rng)
    out["salario_dist"] = m.orden_distancias(e[ms], valor[ms].astype(float), rng=rng)
    mn = cat == "nacimiento"
    out["nac_pc1"] = m.orden_pc1(e[mn], valor[mn].astype(float), n_perm, rng)
    out["nac_vec"] = m.consistencia_ordinal(e[mn], valor[mn].astype(float), k=4, ventana=3, n_perm=n_perm, rng=rng)
    out["nac_dist"] = m.orden_distancias(e[mn], valor[mn].astype(float), rng=rng)
    for nombre in ("tamaño", "antigüedad"):
        mm = cat == nombre
        orden = np.argsort(valor[mm])
        out[f"{nombre}_pares"] = m.orden_pares(e[mm][orden])
    mr = cat == "rama"
    out["rama_letra"] = m.pureza(e[mr], t["letra"].to_numpy()[mr], K_RAMA, n_perm, rng)
    out["rama_r32"] = m.pureza(e[mr], t["r32"].to_numpy()[mr], K_RAMA, n_perm, rng)
    mp = cat == "provincia"
    out["prov_region"] = m.pureza(e[mp], t["region"].to_numpy()[mp], K_PROV, n_perm, rng)
    grados = m.grado_entrada(e, K_HUB)
    grupos = {c_: np.flatnonzero(cat == c_) for c_ in figuras.NOMBRES_CATEGORIA}
    out["hub"] = m.resumen_hubness(grados, grupos)
    out["grados"] = grados
    return out


def vecinos_tabla(e: np.ndarray, c, tokens: list[str], k: int = 5) -> str:
    t = c.tabla
    nombres = t["token"].to_list()
    desc = dict(zip(nombres, t["descripcion"].to_list()))
    nb = m.vecinos(e, k)
    filas = ["| Token | Vecinos más cercanos (coseno) |", "|---|---|"]
    for tok in tokens:
        i = nombres.index(tok)
        etiqueta = f"{tok} ({desc[tok][:40]})" if desc[tok] else tok
        vec = ", ".join(f"{nombres[j]}" for j in nb[i])
        filas.append(f"| {etiqueta} | {vec} |")
    return "\n".join(filas)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-perm", type=int, default=1000, help="permutaciones de etiquetas por medida")
    ap.add_argument("--n-iter", type=int, default=5000, help="iteraciones del test de permutación entre modelos")
    args = ap.parse_args()
    t0 = time.time()
    salida = INFORMES / "fase5"
    salida.mkdir(parents=True, exist_ok=True)
    c = cargar_conceptos()
    emb = {n: matriz_embeddings(r)[N_ESPECIALES:] for n, r in MODELOS.items()}
    assert all(e.shape[0] == len(c) for e in emb.values())
    rng = np.random.default_rng(0)

    res = {n: metricas_modelo(e, c, args.n_perm, np.random.default_rng(1)) for n, e in emb.items()}
    print(f"medidas por modelo listas ({time.time() - t0:.0f} s)", flush=True)

    # ---- robustez entre modelos ----
    d = {n: m.distancias_coseno(e) for n, e in emb.items()}
    perm = m.distancias_coseno(m.nulo_columnas_permutadas(emb["s0"], rng))
    gauss = m.distancias_coseno(m.nulo_gaussiano(emb["s0"], rng))
    comparaciones = [("s0 contra s1", d["s0"], d["s1"]), ("s0 contra s2", d["s0"], d["s2"]),
                     ("s1 contra s2", d["s1"], d["s2"]), ("s0 contra ep30", d["s0"], d["ep30"]),
                     ("s0 contra columnas permutadas", d["s0"], perm), ("s0 contra gaussiano", d["s0"], gauss)]
    rob = [(nombre, m.test_permutacion(a, b, args.n_iter, rng)) for nombre, a, b in comparaciones]
    p_aj = m.fdr_bh(np.array([r["p"] for _, r in rob]))
    print(f"robustez global lista ({time.time() - t0:.0f} s)", flush=True)

    cat = c.tabla["categoria"].to_numpy()
    por_cat = {}
    for categoria in ("salario", "nacimiento", "rama", "provincia"):
        idx = np.flatnonzero(cat == categoria)
        dd = {n: m.distancias_coseno(emb[n][idx]) for n in SEMILLAS}
        por_cat[categoria] = {f"{a} contra {b}": m.test_permutacion(dd[a], dd[b], 2000, rng)
                              for a, b in (("s0", "s1"), ("s0", "s2"), ("s1", "s2"))}
    print(f"robustez por categoría lista ({time.time() - t0:.0f} s)", flush=True)

    # ---- figuras ----
    xy = figuras.proyectar(emb["s0"])
    figuras.mapa_categorias(xy, c.tabla, salida / "mapa_conceptos.png", "Espacio de conceptos (PaCMAP, modelo s0)")
    figuras.paneles(xy, c.tabla, salida / "mapa_paneles.png", "Estructura por categoría (PaCMAP, modelo s0)")
    figuras.dispersion_modelos({**d, "permutado": perm}, [("s0", "s1"), ("s0", "s2"), ("s1", "s2"), ("s0", "permutado")],
                               salida / "dispersion_modelos.png", rng=rng)
    figuras.hubness({n: res[n]["grados"] for n in SEMILLAS}, salida / "hubness.png", K_HUB)

    # ---- tablas ----
    L = ["# Resultados del análisis de conceptos (generado por `python -m labor2vec.analisis.conceptos`)\n"]
    L.append(f"Tokens analizados: {len(c)} (sin los 10 especiales). Nulos: {args.n_perm} permutaciones de etiquetas por "
             f"medida; {args.n_iter} iteraciones en el test de permutación entre modelos. Modelos: `s0`, `s1`, `s2` = "
             "150 épocas con las semillas 0, 1 y 2; `ep30` = 30 épocas (semilla 0).\n")
    L.append("## Orden de las variables ordinales\n")
    L.append("| Medida | " + " | ".join(MODELOS) + " |\n|---|" + "---|" * len(MODELOS))

    def fila(nombre, fn):
        L.append(f"| {nombre} | " + " | ".join(fn(res[n]) for n in MODELOS) + " |")

    fila("Salario (percentil 1-100): \\|Spearman\\| con la 1ª componente principal (p)",
         lambda r: f"{r['salario_pc1']['rho']:.3f} ({_fmt_p(r['salario_pc1']['p'])})")
    fila("Salario: proporción de los 4 vecinos a ≤ 5 percentiles (nulo)",
         lambda r: f"{r['salario_vec']['prop']:.3f} ({r['salario_vec']['nulo']:.3f})")
    fila("Salario: Spearman entre la distancia coseno de cada par y \\|Δpercentil\\| (nulo)",
         lambda r: f"{r['salario_dist']['rho']:.3f} ({r['salario_dist']['nulo']:.3f})")
    fila("Año de nacimiento: \\|Spearman\\| con la 1ª componente (p)",
         lambda r: f"{r['nac_pc1']['rho']:.3f} ({_fmt_p(r['nac_pc1']['p'])})")
    fila("Año de nacimiento: proporción de los 4 vecinos a ≤ 3 años (nulo)",
         lambda r: f"{r['nac_vec']['prop']:.3f} ({r['nac_vec']['nulo']:.3f})")
    fila("Año de nacimiento: Spearman entre la distancia coseno de cada par y \\|Δaño\\| (nulo)",
         lambda r: f"{r['nac_dist']['rho']:.3f} ({r['nac_dist']['nulo']:.3f})")
    fila("Tamaño de empresa (4 niveles): Spearman distancia contra \\|Δnivel\\|", lambda r: f"{r['tamaño_pares']['rho']:.2f}")
    fila("Antigüedad de la empresa (4 niveles): idem", lambda r: f"{r['antigüedad_pares']['rho']:.2f}")
    L.append("\n## Pureza de vecindades\n")
    L.append("| Medida | " + " | ".join(MODELOS) + " |\n|---|" + "---|" * len(MODELOS))
    fila(f"Ramas: pureza de los {K_RAMA} vecinos respecto de la letra (nulo; p)",
         lambda r: f"{r['rama_letra']['pureza']:.3f} ({r['rama_letra']['nulo']:.3f}; {_fmt_p(r['rama_letra']['p'])})")
    fila(f"Ramas: pureza respecto del r32 (nulo; p)",
         lambda r: f"{r['rama_r32']['pureza']:.3f} ({r['rama_r32']['nulo']:.3f}; {_fmt_p(r['rama_r32']['p'])})")
    fila(f"Provincias: pureza de los {K_PROV} vecinos respecto de la región (nulo; p)",
         lambda r: f"{r['prov_region']['pureza']:.3f} ({r['prov_region']['nulo']:.3f}; {_fmt_p(r['prov_region']['p'])})")
    L.append("\n## Robustez entre modelos (Spearman entre las distancias coseno de los 509 tokens)\n")
    L.append("| Comparación | Spearman | Media del nulo | p (permutación) | p ajustado (BH) |\n|---|---|---|---|---|")
    for (nombre, r), pa in zip(rob, p_aj):
        L.append(f"| {nombre} | {r['rho']:.3f} | {r['nulo_medio']:.3f} | {_fmt_p(r['p'])} | {_fmt_p(float(pa))} |")
    L.append("\nPor categoría (sólo los tokens de cada una; 2.000 iteraciones):\n")
    L.append("| Categoría | s0 contra s1 | s0 contra s2 | s1 contra s2 |\n|---|---|---|---|")
    for categoria, dic in por_cat.items():
        L.append(f"| {categoria} | " + " | ".join(f"{r['rho']:.3f} (p {_fmt_p(r['p'])})" for r in dic.values()) + " |")
    L.append(f"\n## Hubness (grafo de {K_HUB} vecinos por coseno; grado medio = {K_HUB})\n")
    L.append("| Medida | " + " | ".join(MODELOS) + " |\n|---|" + "---|" * len(MODELOS))
    fila("Asimetría de la distribución del grado de entrada", lambda r: f"{r['hub']['asimetria']:.2f}")
    fila("Grado máximo", lambda r: str(r["hub"]["maximo"]))
    fila("Proporción de tokens con grado 0", lambda r: f"{r['hub']['prop_cero']:.1%}")
    L.append("\nMediana del grado y proporción con grado 0 por categoría (modelo s0):\n")
    L.append("| Categoría | Tokens | Mediana | Máximo | Grado 0 |\n|---|---|---|---|---|")
    for nombre, g in res["s0"]["hub"]["por_grupo"].items():
        L.append(f"| {figuras.NOMBRES_CATEGORIA[nombre]} | {g['n']} | {g['mediana']:.0f} | {g['max']} | {g['prop_cero']:.0%} |")
    # vecinos de ejemplo
    ramas = c.tabla.filter(c.tabla["categoria"] == "rama").sort("frecuencia", descending=True)["token"].to_list()[:8]
    ejemplos = ramas + ["PROV_06", "PROV_02", "REM_000", "REM_050", "REM_100", "NAC_1980", "TAM_1", "SEXO_M"]
    L.append("\n## Vecinos de ejemplo (modelo s0)\n")
    L.append(vecinos_tabla(emb["s0"], c, [t_ for t_ in ejemplos if t_ in c.tabla["token"].to_list()]))
    # el vecino más cercano de REM_000
    nb = m.vecinos(emb["s0"], 3)
    nombres = c.tabla["token"].to_list()
    L.append("\nVecinos de `REM_000` (remuneración cero) por modelo: " + "; ".join(
        f"{n}: " + ", ".join(nombres[j] for j in m.vecinos(emb[n], 3)[nombres.index("REM_000")]) for n in MODELOS) + ".")
    (salida / "resultados.md").write_text("\n".join(L) + "\n")
    print(f"listo en {time.time() - t0:.0f} s; {salida}")


if __name__ == "__main__":
    main()
