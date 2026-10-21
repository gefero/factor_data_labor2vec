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

Variantes de la fase 8 (plans/PLAN_fase_8.md), derivadas de las tablas anteriores (`--variantes` las reconstruye sin
volver a leer el panel):
- T2c `t2c.parquet`: cambio de quintil entre 2015 y 2019: 0 baja, 1 igual, 2 sube (ordinal). Cohorte de T2.
- T3d `t3d.parquet`: destino de la letra entre quienes la cambian (`y` de T3 = 1): las 5 letras de destino más frecuentes
  en train (0 la más frecuente) y 5 = otras. Nominal.
- T4c `t4c.parquet`: 1 si el tramo de tamaño de 2019 difiere del de 2015. Cohorte de T4.

Continuidad (fase 8): `continuidad.parquet` marca, para cada persona de la cohorte de T1, si su relación principal de 2015
**continúa después de 2015-12** (ahí el «último valor» de rama, provincia y antigüedad puede venir del futuro) o terminó en
2015-12 o antes (sin fuga posible). `--continuidad` lo calcula leyendo el panel.

    python -m labor2vec.tareas.derivadas [--variantes | --continuidad]
"""

from pathlib import Path

import numpy as np
import polars as pl

from labor2vec.tareas.t1 import ANIO_CORTE, CORTE, _ventana
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


N_DESTINOS = 5


def sql_continuidad(panel: Path, t1: Path) -> str:
    """Una fila por persona de la cohorte: `ultimo_mes` de su relación principal de 2015 y `continua` (ultimo_mes > 2015-12).
    Relación principal = la de más meses en 2015 (desempate por mayor remuneración, luego por id), como en `sql_estado`."""
    a0, a1 = _ventana(ANIO_CORTE)
    return f"""
WITH coh AS (SELECT id_trabajador FROM '{t1}'),
rel AS (
    SELECT p.id_trabajador, p.id_relacion,
           count(*) FILTER (WHERE p.mes_idx BETWEEN {a0} AND {a1}) AS meses,
           coalesce(sum(p.rem_tot) FILTER (WHERE p.mes_idx BETWEEN {a0} AND {a1}), 0) AS rem,
           max(p.mes_idx) AS ultimo_mes
    FROM '{panel}' p JOIN coh USING (id_trabajador) GROUP BY ALL),
principal AS (
    SELECT id_trabajador, ultimo_mes FROM (
        SELECT *, row_number() OVER (PARTITION BY id_trabajador ORDER BY meses DESC, rem DESC, id_relacion) AS rk
        FROM rel WHERE meses > 0) WHERE rk = 1)
SELECT id_trabajador, ultimo_mes, ultimo_mes > {CORTE} AS continua FROM principal ORDER BY id_trabajador
"""


def continuidad(panel: Path = PANEL, t1: Path = T1, salida: Path = TAREAS) -> pl.DataFrame:
    df = conectar().execute(sql_continuidad(panel, t1)).pl()
    Path(salida).mkdir(parents=True, exist_ok=True)
    df.write_parquet(Path(salida) / "continuidad.parquet")
    return df


def variantes(t2: pl.DataFrame, t3: pl.DataFrame, t4: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """T2c, T3d y T4c a partir de las tablas de T2, T3 y T4 (mismos splits)."""
    t2c = t2.with_columns((pl.col("y").cast(pl.Int16) - pl.col("q2015")).sign().cast(pl.Int8).add(1).alias("y_q"),
                          pl.col("y").alias("q2019")).drop("y").rename({"y_q": "y"})
    cambian = t3.filter(pl.col("y") == 1)
    top = (cambian.filter(pl.col("split") == "train").group_by("letra2019").len()
           .sort(["len", "letra2019"], descending=[True, False])["letra2019"].to_list()[:N_DESTINOS])
    codigo = {int(letra): k for k, letra in enumerate(top)}
    t3d = cambian.with_columns(
        pl.col("letra2019").map_elements(lambda x: codigo.get(int(x), N_DESTINOS), return_dtype=pl.Int8).alias("y_d")
    ).drop("y").rename({"y_d": "y"})
    t4c = t4.with_columns((pl.col("y") != 1).cast(pl.Int8).alias("y_c"), pl.col("y").alias("direccion")).drop("y").rename(
        {"y_c": "y"})
    return {"t2c": t2c, "t3d": t3d, "t4c": t4c}


def guardar(out: dict[str, pl.DataFrame], salida: Path) -> None:
    Path(salida).mkdir(parents=True, exist_ok=True)
    for nombre, df in out.items():
        df.write_parquet(Path(salida) / f"{nombre}.parquet")


def construir(panel: Path = PANEL, t1: Path = T1, salida: Path = TAREAS) -> dict[str, pl.DataFrame]:
    estado = conectar().execute(sql_estado(panel, t1)).pl()
    out = tablas(estado)
    out |= variantes(out["t2"], out["t3"], out["t4"])
    guardar(out, salida)
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
    import sys

    if "--continuidad" in sys.argv:
        df = continuidad()
        print(f"{df.height} personas; continúan después de 2015-12: {df['continua'].mean():.3f}")
    elif "--variantes" in sys.argv:
        base = {t: pl.read_parquet(TAREAS / f"{t}.parquet") for t in ("t2", "t3", "t4")}
        nuevas = variantes(**base)
        guardar(nuevas, TAREAS)
        print(resumen(base | nuevas))
    else:
        print(resumen(construir()))
