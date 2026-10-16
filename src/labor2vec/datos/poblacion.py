"""Tablas de trabajadores y relaciones, exclusiones y splits.

- trabajadores.parquet: una fila por trabajador, con atributos fijos, resumen de su historia y
  la marca de exclusión (sin sexo o sin año de nacimiento; decisión del 2026-10-15).
- relaciones.parquet: una fila por (trabajador, relación), con los atributos fijos de la relación
  (provincia, ramas y antigüedad de la empresa: "último valor" según el manual), su duración,
  intermitencias y censura.
- splits.parquet: asignación fija 70/15/15 (train/val/test) por trabajador entre los no excluidos.

    python -m labor2vec.datos.poblacion
"""

import numpy as np
import polars as pl

from labor2vec.utiles.duck import conectar
from labor2vec.utiles.rutas import PANEL_BASE, RELACIONES, SPLITS, TRABAJADORES

SEMILLA = 20261015
PROPORCIONES = {"train": 0.70, "val": 0.15, "test": 0.15}

SQL_TRABAJADORES = f"""
SELECT
    id_trabajador,
    any_value(sexo)                     AS sexo,
    any_value(fnacim)                   AS fnacim,
    min(tiempo)                         AS primer_mes,
    max(tiempo)                         AS ultimo_mes,
    count(DISTINCT tiempo)::SMALLINT    AS n_meses,
    count(*)::INTEGER                   AS n_filas,
    count(DISTINCT id_relacion)::SMALLINT AS n_relaciones,
    CASE WHEN any_value(sexo) IS NULL AND any_value(fnacim) IS NULL THEN 'sin_sexo_ni_fnacim'
         WHEN any_value(sexo) IS NULL THEN 'sin_sexo'
         WHEN any_value(fnacim) IS NULL THEN 'sin_fnacim' END AS motivo_exclusion
FROM '{PANEL_BASE}'
GROUP BY id_trabajador
ORDER BY id_trabajador
"""

SQL_RELACIONES = f"""
WITH m AS (
    SELECT id_trabajador, id_relacion, mes_idx, provi, letra, r32, r34, tr_anio_ini,
           mes_idx - lag(mes_idx) OVER (PARTITION BY id_trabajador, id_relacion ORDER BY mes_idx) AS salto
    FROM '{PANEL_BASE}'
)
SELECT
    id_trabajador,
    id_relacion,
    any_value(provi)        AS provi,
    any_value(letra)        AS letra,
    any_value(r32)          AS r32,
    any_value(r34)          AS r34,
    any_value(tr_anio_ini)  AS tr_anio_ini,
    min(mes_idx)            AS inicio_mes_idx,
    max(mes_idx)            AS fin_mes_idx,
    count(*)::SMALLINT      AS n_meses,
    count(*) FILTER (WHERE salto > 1)::SMALLINT AS n_brechas,
    (min(mes_idx) = 0)      AS censura_izq,
    (max(mes_idx) = 311)    AS censura_der
FROM m
GROUP BY id_trabajador, id_relacion
ORDER BY id_trabajador, id_relacion
"""


def construir_splits(trab: pl.DataFrame) -> pl.DataFrame:
    ids = np.sort(trab.filter(pl.col("motivo_exclusion").is_null())["id_trabajador"].to_numpy())
    np.random.default_rng(SEMILLA).shuffle(ids)
    cortes = np.round(np.cumsum(list(PROPORCIONES.values())) * len(ids)).astype(int)[:-1]
    partes = np.split(ids, cortes)
    return pl.concat(
        pl.DataFrame({"id_trabajador": p, "split": nombre}) for nombre, p in zip(PROPORCIONES, partes)
    ).sort("id_trabajador")


def construir() -> None:
    con = conectar()
    trab = con.execute(SQL_TRABAJADORES).pl()
    trab.write_parquet(TRABAJADORES)
    excl = trab["motivo_exclusion"].value_counts().sort("motivo_exclusion")
    print(f"trabajadores: {trab.height:,}; exclusiones:\n{excl}")

    con.execute(f"COPY ({SQL_RELACIONES}) TO '{RELACIONES}' (FORMAT parquet, COMPRESSION zstd)")
    n_rel = con.execute(f"SELECT count(*) FROM '{RELACIONES}'").fetchone()[0]
    print(f"relaciones: {n_rel:,}")

    splits = construir_splits(trab)
    splits.write_parquet(SPLITS)
    print(f"splits:\n{splits['split'].value_counts().sort('split')}")


if __name__ == "__main__":
    construir()
