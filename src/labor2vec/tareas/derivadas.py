"""T2, T3 y T4: etiquetas derivadas del estado del trabajador en 2015 y en 2019 (plans/PLAN_fase_6b.md).

Población: la cohorte de T1 que sigue registrada en 2019 (`y_2019 = 0`), con los mismos splits.
Estado de un trabajador en un año (2015: meses 228-239; 2019: 276-287):
- relación principal = la de más meses registrados en el año (desempate por mayor remuneración, luego por id);
  de ahí salen `letra` y `tramo_empleo`;
- nivel salarial = promedio de `pct_rem_mes` sobre las filas con remuneración positiva.

Tareas (cada una con su parquet en `data/processed/tareas/`, columna `y`):
- T2 `t2.parquet`: quintil (0-4) del nivel salarial de 2019. Los cortes se ajustan sólo con train y se aplican también a 2015
  (`q2015`, la línea de base de persistencia).
- T3 `t3.parquet`: 1 si la letra de 2019 difiere de la de 2015.
- T4 `t4.parquet`: tamaño de la empresa principal respecto de 2015: 0 más chica, 1 igual, 2 más grande
  (`tramo_empleo` 1 = más de 200 empleados: un número menor es una empresa más grande).

    python -m labor2vec.tareas.derivadas
"""

from pathlib import Path

import numpy as np
import polars as pl

from labor2vec.tareas.t1 import ANIO_CORTE, _ventana
from labor2vec.utiles.duck import conectar
from labor2vec.utiles.rutas import PANEL, T1, TAREAS

ANIO_FIN = 2019
QUINTILES = [0.2, 0.4, 0.6, 0.8]


def sql_estado(panel: Path, t1: Path) -> str:
    """Una fila por trabajador de la cohorte que sigue en 2019, con su estado en 2015 y en 2019."""
    (a0, a1), (b0, b1) = _ventana(ANIO_CORTE), _ventana(ANIO_FIN)
    return f"""
WITH coh AS (
    SELECT id_trabajador, split, sexo, fnacim, edad_corte FROM '{t1}' WHERE NOT y_2019),
filas AS (
    SELECT p.id_trabajador, p.id_relacion, p.letra, p.tramo_empleo, p.rem_tot, p.pct_rem_mes,
           CASE WHEN p.mes_idx BETWEEN {a0} AND {a1} THEN {ANIO_CORTE} ELSE {ANIO_FIN} END AS anio
    FROM '{panel}' p JOIN coh USING (id_trabajador)
    WHERE p.mes_idx BETWEEN {a0} AND {a1} OR p.mes_idx BETWEEN {b0} AND {b1}),
rel AS (
    SELECT id_trabajador, anio, id_relacion, max(letra) AS letra, max(tramo_empleo) AS tramo, count(*) AS meses,
           sum(rem_tot) AS rem
    FROM filas GROUP BY ALL),
principal AS (
    SELECT id_trabajador, anio, letra, tramo FROM (
        SELECT *, row_number() OVER (PARTITION BY id_trabajador, anio
                                     ORDER BY meses DESC, rem DESC, id_relacion) AS rk FROM rel) WHERE rk = 1),
sal AS (
    SELECT id_trabajador, anio, avg(pct_rem_mes) FILTER (WHERE pct_rem_mes > 0) AS pct FROM filas GROUP BY ALL)
SELECT c.*,
       a.letra AS letra2015, a.tramo AS tramo2015, b.letra AS letra2019, b.tramo AS tramo2019,
       sa.pct AS pct2015, sb.pct AS pct2019
FROM coh c
LEFT JOIN principal a ON a.id_trabajador = c.id_trabajador AND a.anio = {ANIO_CORTE}
LEFT JOIN principal b ON b.id_trabajador = c.id_trabajador AND b.anio = {ANIO_FIN}
LEFT JOIN sal sa ON sa.id_trabajador = c.id_trabajador AND sa.anio = {ANIO_CORTE}
LEFT JOIN sal sb ON sb.id_trabajador = c.id_trabajador AND sb.anio = {ANIO_FIN}
ORDER BY c.id_trabajador
"""


def cortes_quintiles(estado: pl.DataFrame) -> np.ndarray:
    """Cuantiles 20/40/60/80 % del nivel salarial de 2019, ajustados sólo con train."""
    pct = estado.filter((pl.col("split") == "train") & pl.col("pct2019").is_not_null())["pct2019"].to_numpy()
    return np.quantile(pct, QUINTILES)


def quintil(pct: np.ndarray, cortes: np.ndarray) -> np.ndarray:
    return np.searchsorted(cortes, pct, side="right").astype(np.int8)  # 0..4


def tablas(estado: pl.DataFrame) -> dict[str, pl.DataFrame]:
    base = ["id_trabajador", "split", "sexo", "fnacim", "edad_corte"]
    cortes = cortes_quintiles(estado)
    t2 = estado.filter(pl.col("pct2015").is_not_null() & pl.col("pct2019").is_not_null())
    t2 = t2.with_columns(
        pl.Series("q2015", quintil(t2["pct2015"].to_numpy(), cortes), dtype=pl.Int8),
        pl.Series("y", quintil(t2["pct2019"].to_numpy(), cortes), dtype=pl.Int8)
    ).select(*base, "y", "q2015", "pct2015", "pct2019")

    t3 = estado.filter(pl.col("letra2015").is_not_null() & pl.col("letra2019").is_not_null()).with_columns(
        (pl.col("letra2015") != pl.col("letra2019")).cast(pl.Int8).alias("y")
    ).select(*base, "y", "letra2015", "letra2019")

    t4 = estado.filter((pl.col("tramo2015") > 0) & (pl.col("tramo2019") > 0)).with_columns(
        pl.when(pl.col("tramo2019") > pl.col("tramo2015")).then(0)
        .when(pl.col("tramo2019") < pl.col("tramo2015")).then(2).otherwise(1).cast(pl.Int8).alias("y")
    ).select(*base, "y", "tramo2015", "tramo2019")
    return {"t2": t2, "t3": t3, "t4": t4}


def construir(panel: Path = PANEL, t1: Path = T1, salida: Path = TAREAS) -> dict[str, pl.DataFrame]:
    estado = conectar().execute(sql_estado(panel, t1)).pl()
    out = tablas(estado)
    Path(salida).mkdir(parents=True, exist_ok=True)
    for nombre, df in out.items():
        df.write_parquet(Path(salida) / f"{nombre}.parquet")
    return out


def resumen(out: dict[str, pl.DataFrame]) -> str:
    partes = []
    for nombre, df in out.items():
        por_split = df.group_by("split").agg(pl.len().alias("n")).sort("split")
        dist = df.group_by("y").agg(pl.len().alias("n")).sort("y").with_columns((pl.col("n") / df.height).alias("frac"))
        partes.append(f"{nombre}: n = {df.height:,}\nsplits\n{por_split}\nclases\n{dist}")
    t2 = out["t2"]
    partes.append(f"T2 persistencia (q2015 = y): {(t2['q2015'] == t2['y']).mean():.3f}; "
                  f"correlación de niveles 2015-2019: {np.corrcoef(t2['pct2015'], t2['pct2019'])[0, 1]:.3f}")
    return "\n\n".join(partes)


if __name__ == "__main__":
    print(resumen(construir()))
