"""Panel sintético compartido por las pruebas de las fases 2 y 3."""

import numpy as np
import polars as pl
import pytest

from labor2vec.tokens import corpus as corpus_mod
from labor2vec.tokens.vocabulario import Vocabulario

UMBRAL_RAMA, UMBRAL_NAC = 5, 2


def _trabajadores(n=60):
    """Trabajadores 1..n; el 1 es el caso de prueba (dos relaciones simultáneas y una brecha)."""
    filas, rels, trab, splits = [], [], [], []
    rng = np.random.default_rng(0)
    for t in range(1, n + 1):
        split = "train" if t <= 40 else ("val" if t <= 50 else "test")
        splits.append((t, split))
        trab.append((t, 1 + t % 2, 1970 + t % 20))
        if t == 1:
            rel = {1: [0, 1, 2, 6], 2: [1, 2, 3]}
        else:
            rel = {1: list(range(int(rng.integers(0, 5)), int(rng.integers(8, 40))))}
        for r, meses in rel.items():
            rels.append((t, r, meses[0]))
            for m in meses:
                # rama 111 y 112 frecuentes; 113 rara (sólo el trabajador 3 en train)
                r34 = 113 if t == 3 else (111 if t % 2 else 112)
                filas.append((t, r, m, (m * 7 + t) % 101, 1 + (m + r) % 4, 1 + r % 4, 1 + t % 3, 11, r34,
                              1970 + t % 20, 1 + t % 2, split))
    cols = ["id_trabajador", "id_relacion", "mes_idx", "pct_rem_mes", "tramo_empleo", "tr_anio_ini", "provi", "r32",
            "r34", "fnacim", "sexo", "split"]
    panel = pl.DataFrame(filas, schema=cols, orient="row").with_columns(
        pl.col("r32").cast(pl.Int16), pl.col("r34").cast(pl.Int16))
    return (panel, pl.DataFrame(rels, schema=["id_trabajador", "id_relacion", "inicio_mes_idx"], orient="row"),
            pl.DataFrame(trab, schema=["id_trabajador", "sexo", "fnacim"], orient="row"),
            pl.DataFrame(splits, schema=["id_trabajador", "split"], orient="row"))


@pytest.fixture(scope="module")
def mundo(tmp_path_factory):
    d = tmp_path_factory.mktemp("mler")
    panel, rels, trab, splits = _trabajadores()
    rutas = {k: d / f"{k}.parquet" for k in ("panel", "relaciones", "trabajadores", "splits")}
    for k, df in zip(rutas, (panel, rels, trab, splits)):
        df.write_parquet(rutas[k])
    salida = d / "corpus"
    totales = corpus_mod.construir(salida, tramos=3, **rutas)
    vocab = Vocabulario.ajustar(rutas["panel"], umbral_rama=UMBRAL_RAMA, umbral_nac=UMBRAL_NAC)
    vocab.guardar(salida / "vocabulario.parquet", salida / "ramas.parquet")
    return dict(panel=panel, salida=salida, totales=totales, vocab=vocab, rutas=rutas)
