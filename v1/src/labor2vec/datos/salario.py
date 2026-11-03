"""Salario relativo: percentil de la remuneración dentro de cada mes → panel.parquet.

Las remuneraciones son nominales con inflación fuerte (la mediana pasa de ~557 en 1996 a ~73.500
en 2021), así que se ubican respecto de la distribución de su propio mes. Los cortes (percentiles
1 a 99 de rem_tot > 0) se calculan **sólo con el split de entrenamiento**, como el paper ajusta sus
bins sobre train, y se aplican a todas las filas.

pct_rem_mes: 0 si rem_tot = 0 (licencia, según el manual); 1..100 si rem_tot > 0, con
1 + (cantidad de cortes <= rem_tot), acotado a 100.

panel.parquet = panel_base + split (nulo para los excluidos) + pct_rem_mes.

    python -m labor2vec.datos.salario
"""

from labor2vec.utiles.duck import conectar
from labor2vec.utiles.rutas import CORTES_SALARIO, PANEL, PANEL_BASE, SPLITS

PROBS = [round(p / 100, 2) for p in range(1, 100)]


def construir() -> None:
    con = conectar()
    con.execute(f"""
        COPY (
            SELECT p.tiempo, quantile_cont(p.rem_tot, {PROBS}) AS cortes, count(*) AS n_train
            FROM '{PANEL_BASE}' p JOIN '{SPLITS}' s USING (id_trabajador)
            WHERE s.split = 'train' AND p.rem_tot > 0
            GROUP BY p.tiempo ORDER BY p.tiempo
        ) TO '{CORTES_SALARIO}' (FORMAT parquet)""")
    n_meses, n_min = con.execute(f"SELECT count(*), min(n_train) FROM '{CORTES_SALARIO}'").fetchone()
    print(f"cortes: {n_meses} meses; mínimo {n_min:,} remuneraciones de train por mes")

    # Tabla larga (tiempo, corte, k): k = cantidad de cortes <= corte (con cortes repetidos, el mayor k).
    # El ASOF JOIN toma, para cada fila, el mayor corte <= rem_tot de su mes: pct = k + 1.
    # (Antes se pegaba la lista de 99 cortes a cada una de las 49 M filas y se filtraba con una lambda:
    # el mismo resultado, pero con tanta memoria que tildó la máquina.)
    con.execute(f"""
        CREATE TEMP TABLE cortes_largo AS
        SELECT tiempo, corte, max(k) AS k FROM (
            SELECT tiempo, unnest(cortes) AS corte, unnest(range(1, 100)) AS k FROM '{CORTES_SALARIO}')
        GROUP BY tiempo, corte""")
    tmp = PANEL.with_suffix(".tmp")
    con.execute(f"""
        COPY (
            SELECT p.*, s.split,
                   CASE WHEN p.rem_tot <= 0 THEN 0 ELSE coalesce(c.k, 0) + 1 END::TINYINT AS pct_rem_mes
            FROM '{PANEL_BASE}' p
            LEFT JOIN '{SPLITS}' s USING (id_trabajador)
            ASOF LEFT JOIN cortes_largo c ON p.tiempo = c.tiempo AND p.rem_tot >= c.corte
            ORDER BY p.id_trabajador, p.tiempo, p.id_relacion
        ) TO '{tmp}' (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 1000000)""")
    tmp.rename(PANEL)
    # control: en train, cada percentil debería tener ~1 % de las remuneraciones positivas de cada mes
    desvio = con.execute(f"""
        SELECT max(abs(f - 0.01)) FROM (
            SELECT count(*) / sum(count(*)) OVER (PARTITION BY tiempo) AS f
            FROM '{PANEL}' WHERE split = 'train' AND pct_rem_mes > 0
            GROUP BY tiempo, pct_rem_mes)""").fetchone()[0]
    n = con.execute(f"SELECT count(*) FROM '{PANEL}'").fetchone()[0]
    print(f"panel: {n:,} filas; máximo desvío de la proporción por percentil en train: {desvio:.4f}")


if __name__ == "__main__":
    construir()
