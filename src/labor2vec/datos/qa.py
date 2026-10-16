"""Reporte de calidad de la MLER procesada → informes/qa_mler.md.

Sólo agregados (ningún dato individual): faltantes, remuneraciones atípicas, largos de las
historias, relaciones, simultaneidad, censura, edades implícitas, categorías raras de rama,
balance de los splits y largo estimado de las secuencias para la fase 2.

    python -m labor2vec.datos.qa
"""

from datetime import datetime

import polars as pl

from labor2vec.utiles.duck import conectar
from labor2vec.utiles.rutas import INFORMES, PANEL, RELACIONES, TRABAJADORES

CUANTILES = [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 1.0]


SIN_SEPARADOR = {"anio", "desde", "hasta", "anio_nac_medio"}  # años y fechas AAAAMM


def tabla_md(df: pl.DataFrame, decimales: int = 2) -> str:
    def fmt(v, col):
        if v is None:
            return "—"
        sep = "" if col in SIN_SEPARADOR else ","
        if isinstance(v, float):
            return f"{v:{sep}.{decimales}f}"
        if isinstance(v, int):
            return f"{v:{sep}}"
        return str(v)
    filas = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * df.width]
    filas += ["| " + " | ".join(fmt(v, c) for v, c in zip(r, df.columns)) + " |" for r in df.iter_rows()]
    return "\n".join(filas)


def cuantiles_sql(expr: str) -> str:
    return ", ".join(f"quantile_disc({expr}, {q}) AS \"p{int(q * 100)}\"" for q in CUANTILES)


def generar() -> str:
    con = conectar()
    P, T, R = f"'{PANEL}'", f"'{TRABAJADORES}'", f"'{RELACIONES}'"
    q = lambda sql: con.execute(sql).pl()  # noqa: E731
    s = [
        "# Reporte de calidad de la MLER",
        "",
        f"Generado por `python -m labor2vec.datos.qa` ({datetime.now():%Y-%m-%d %H:%M}). Sólo agregados.",
        "Los códigos 0 (\"Sin definir\") de provi, letra, r32, r34, tramo_empleo y sexo ya están como nulos.",
        "",
    ]

    s += ["## 1. Tamaño", "", tabla_md(q(f"""
        SELECT count(*) AS filas, count(DISTINCT id_trabajador) AS trabajadores,
               (SELECT count(*) FROM {R}) AS relaciones, count(DISTINCT tiempo) AS meses,
               min(tiempo) AS desde, max(tiempo) AS hasta FROM {P}""")), ""]

    s += ["## 2. Faltantes (% de filas)", "", tabla_md(q(f"""
        SELECT anio,
               round(100 * avg((provi IS NULL)::INT), 3) AS provi,
               round(100 * avg((letra IS NULL)::INT), 3) AS letra_r32_r34,
               round(100 * avg((tramo_empleo IS NULL)::INT), 3) AS tramo_empleo,
               round(100 * avg((tr_anio_ini IS NULL)::INT), 3) AS tr_anio_ini,
               round(100 * avg((sexo IS NULL)::INT), 3) AS sexo,
               round(100 * avg((fnacim IS NULL)::INT), 3) AS fnacim
        FROM {P} GROUP BY anio ORDER BY anio"""), 3), ""]

    s += ["## 3. Remuneraciones", "",
          "Nominales. `rem_tot = 0` corresponde en general a licencias (manual). "
          "Los salarios mayores al P98 por rama están microagregados y todos tienen ruido de ±3 %.", "",
          tabla_md(q(f"""
        SELECT anio, count(*) FILTER (WHERE rem_tot = 0) AS rem_0,
               count(*) FILTER (WHERE rem_tot > 0 AND rem_tot < 30) AS rem_menor_30,
               count(*) FILTER (WHERE rem_tot > 1e7) AS rem_mayor_10M,
               round(median(rem_tot) FILTER (WHERE rem_tot > 0), 0) AS mediana,
               round(quantile_disc(rem_tot, 0.98) FILTER (WHERE rem_tot > 0), 0) AS p98,
               round(max(rem_tot), 0) AS maximo
        FROM {P} GROUP BY anio ORDER BY anio"""), 0), ""]

    s += ["## 4. Historias por trabajador", "", "Elegibles (sin exclusiones):", "",
          tabla_md(q(f"""
        SELECT 'filas' AS medida, {cuantiles_sql('n_filas')} FROM {T} WHERE motivo_exclusion IS NULL
        UNION ALL SELECT 'meses con registro', {cuantiles_sql('n_meses')} FROM {T} WHERE motivo_exclusion IS NULL
        UNION ALL SELECT 'relaciones', {cuantiles_sql('n_relaciones')} FROM {T} WHERE motivo_exclusion IS NULL
        """)), "", "Exclusiones:", "",
          tabla_md(q(f"""SELECT coalesce(motivo_exclusion, '(elegible)') AS motivo, count(*) AS trabajadores
                         FROM {T} GROUP BY 1 ORDER BY 2 DESC""")), ""]

    s += ["## 5. Relaciones laborales", "", tabla_md(q(f"""
        SELECT 'meses por relación' AS medida, {cuantiles_sql('n_meses')} FROM {R}
        UNION ALL SELECT 'brechas por relación', {cuantiles_sql('n_brechas')} FROM {R}""")), "",
          tabla_md(q(f"""
        SELECT count(*) AS relaciones,
               count(*) FILTER (WHERE n_brechas > 0) AS con_brechas,
               count(*) FILTER (WHERE censura_izq) AS censura_izq_1996_01,
               count(*) FILTER (WHERE censura_der) AS censura_der_2021_12,
               count(*) FILTER (WHERE censura_izq AND censura_der) AS ambas
        FROM {R}""")), ""]

    s += ["## 6. Relaciones simultáneas", "", "Cantidad de relaciones por (trabajador, mes):", "",
          tabla_md(q(f"""
        SELECT n AS relaciones_en_el_mes, count(*) AS trabajador_meses,
               round(100 * count(*) / sum(count(*)) OVER (), 3) AS pct
        FROM (SELECT count(*) AS n FROM {P} GROUP BY id_trabajador, tiempo)
        GROUP BY n ORDER BY n LIMIT 10"""), 3), ""]

    s += ["## 7. Edad implícita (año − año de nacimiento)", "", tabla_md(q(f"""
        SELECT min(anio - fnacim) AS minima, max(anio - fnacim) AS maxima,
               count(*) FILTER (WHERE anio - fnacim < 16) AS filas_menor_16,
               count(*) FILTER (WHERE anio - fnacim < 18) AS filas_menor_18,
               count(*) FILTER (WHERE anio - fnacim > 75) AS filas_mayor_75,
               {cuantiles_sql('anio - fnacim')}
        FROM {P} WHERE fnacim IS NOT NULL""")), ""]

    s += ["## 8. Ramas (r34): categorías raras", "",
          "Trabajadores elegibles de train por código r34. Los códigos con pocos casos irían a `[UNK]` en la fase 2.", "",
          tabla_md(q(f"""
        WITH c AS (SELECT r34, count(DISTINCT id_trabajador) AS n FROM {P}
                   WHERE split = 'train' AND r34 IS NOT NULL GROUP BY r34)
        SELECT count(*) AS codigos, count(*) FILTER (WHERE n < 10) AS menos_de_10,
               count(*) FILTER (WHERE n < 50) AS menos_de_50, count(*) FILTER (WHERE n < 100) AS menos_de_100,
               min(n) AS minimo, median(n) AS mediana, max(n) AS maximo FROM c""")), ""]

    s += ["## 9. Balance de los splits", "", tabla_md(q(f"""
        SELECT s.split, count(*) AS trabajadores,
               round(100 * avg((t.sexo = 1)::INT), 2) AS pct_mujeres,
               round(avg(t.fnacim), 2) AS anio_nac_medio,
               round(avg(t.n_meses), 2) AS meses_medio,
               round(avg(t.n_relaciones), 3) AS relaciones_medio
        FROM {T} t JOIN '{PANEL.parent / "splits.parquet"}' s USING (id_trabajador)
        GROUP BY s.split ORDER BY s.split""")), ""]

    s += ["## 10. Largo estimado de las secuencias (insumo de la fase 2)", "",
          "Variante A del plan: 5 tokens por evento + `[SEP]`, más 4 de fondo (`[CLS]`, sexo, año, `[SEP]`).", "",
          tabla_md(q(f"""
        SELECT {cuantiles_sql('4 + 6 * n_filas')},
               round(100 * avg((4 + 6 * n_filas > 2560)::INT), 2) AS pct_mayor_2560,
               round(100 * avg((4 + 6 * n_filas > 1024)::INT), 2) AS pct_mayor_1024,
               round(100 * avg((4 + 6 * n_filas > 512)::INT), 2) AS pct_mayor_512
        FROM {T} WHERE motivo_exclusion IS NULL""")), ""]
    return "\n".join(s)


if __name__ == "__main__":
    INFORMES.mkdir(exist_ok=True)
    salida = INFORMES / "qa_mler.md"
    salida.write_text(generar())
    print(f"reporte en {salida}")
