"""T1: salida del empleo registrado. Cohorte y etiquetas (plans/PLAN_fase_6.md, sección 6.1).

Cohorte: trabajadores de 25 a 55 años al 31-12-2015 con los 12 meses de 2015 registrados.
Etiquetas (1 = sin ningún mes registrado en el año): `y_2019` (la principal, a 4 años del corte) y
`y_2021` (prueba extra con el shock del COVID). El modelo sólo ve la historia hasta el corte.

    python -m labor2vec.tareas.t1
"""

from pathlib import Path

import polars as pl

from labor2vec.utiles.duck import conectar
from labor2vec.utiles.rutas import PANEL, SPLITS, T1

CORTE = 239  # mes_idx de 2015-12 (0 = 1996-01)
ANIO_CORTE = 2015
EDAD_MIN, EDAD_MAX = 25, 55
MESES_MIN = 12
HORIZONTES = {"y_2019": 2019, "y_2021": 2021}
TRAMOS_EDAD = [(25, 34), (35, 44), (45, 55)]


def _ventana(anio: int) -> tuple[int, int]:
    """mes_idx inicial y final de un año calendario (el corte es diciembre de ANIO_CORTE)."""
    fin = CORTE + 12 * (anio - ANIO_CORTE)
    return fin - 11, fin


def sql_cohorte(panel: Path, splits: Path) -> str:
    a0, a1 = _ventana(ANIO_CORTE)
    flags = ",\n    ".join(
        f"NOT bool_or(mes_idx BETWEEN {_ventana(a)[0]} AND {_ventana(a)[1]}) AS {nombre}"
        for nombre, a in HORIZONTES.items())
    return f"""
WITH t AS (
    SELECT id_trabajador, any_value(sexo) AS sexo, any_value(fnacim) AS fnacim,
           count(DISTINCT mes_idx) FILTER (WHERE mes_idx BETWEEN {a0} AND {a1}) AS meses_ultimo_anio,
           {flags}
    FROM '{panel}'
    GROUP BY id_trabajador
)
SELECT t.id_trabajador, s.split, t.sexo::TINYINT AS sexo, t.fnacim::SMALLINT AS fnacim,
       ({ANIO_CORTE} - t.fnacim)::TINYINT AS edad_corte,
       {", ".join(f"t.{n}" for n in HORIZONTES)}
FROM t JOIN '{splits}' s USING (id_trabajador)
WHERE t.meses_ultimo_anio >= {MESES_MIN} AND ({ANIO_CORTE} - t.fnacim) BETWEEN {EDAD_MIN} AND {EDAD_MAX}
ORDER BY t.id_trabajador
"""


def construir(panel: Path = PANEL, splits: Path = SPLITS, salida: Path = T1) -> pl.DataFrame:
    """Calcula la cohorte con sus etiquetas y la escribe en parquet."""
    con = conectar()
    df = con.execute(sql_cohorte(panel, splits)).pl()
    Path(salida).parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(salida)
    return df


def resumen(df: pl.DataFrame) -> str:
    """Informe de control: tamaño y tasas por split, sexo y tramo de edad."""
    def tabla(clave: pl.Expr, nombre: str) -> pl.DataFrame:
        return (df.group_by(clave.alias(nombre)).agg(
            pl.len().alias("n"), *(pl.col(y).mean().alias(f"tasa_{y}") for y in HORIZONTES)).sort(nombre))
    tramo = pl.col("edad_corte").map_elements(
        lambda e: next(f"{a}-{b}" for a, b in TRAMOS_EDAD if a <= e <= b), return_dtype=pl.String)
    total = df.select(pl.len().alias("n"), *(pl.col(y).mean().alias(f"tasa_{y}") for y in HORIZONTES))
    partes = [("total", total), ("split", tabla(pl.col("split"), "split")),
              ("sexo (1 mujer, 2 varón)", tabla(pl.col("sexo"), "sexo")), ("edad", tabla(tramo, "edad"))]
    return "\n\n".join(f"{t}\n{d}" for t, d in partes)


if __name__ == "__main__":
    print(resumen(construir()))
