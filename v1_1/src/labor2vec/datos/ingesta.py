"""Ingesta: MLER.csv → data/processed/panel_base.parquet.

Una fila por (trabajador, relación, mes), con tipos explícitos, ordenada por
(id_trabajador, tiempo, id_relacion). Los códigos 0 ("Sin definir" / "Sin asignar" según los
descriptores del MTEySS) se pasan a nulo en provi, letra, r32, r34, tramo_empleo y sexo.
`pondera` se descarta: es constante (diseño aleatorio simple al 3 %).

    python -m labor2vec.datos.ingesta [--forzar]
"""

import argparse
import hashlib
import json
import time
from datetime import datetime

from labor2vec.utiles.duck import conectar
from labor2vec.utiles.rutas import CSV_MLER, META_INGESTA, PANEL_BASE, PROCESADOS

TIPOS_CSV = {
    "id_trabajador": "INTEGER", "id_relacion": "SMALLINT", "tiempo": "INTEGER", "rem_tot": "DOUBLE",
    "tramo_empleo": "TINYINT", "tr_anio_ini": "TINYINT", "provi": "SMALLINT", "r34": "SMALLINT",
    "r32": "SMALLINT", "letra": "TINYINT", "fnacim": "SMALLINT", "sexo": "TINYINT", "pondera": "DOUBLE",
}

CONSULTA = """
SELECT
    id_trabajador,
    id_relacion,
    tiempo,
    (tiempo // 100)::SMALLINT                               AS anio,
    (tiempo % 100)::TINYINT                                 AS mes,
    ((tiempo // 100 - 1996) * 12 + tiempo % 100 - 1)::SMALLINT AS mes_idx,
    make_date(tiempo // 100, tiempo % 100, 15)              AS fecha,
    date_diff('day', DATE '1996-01-01', make_date(tiempo // 100, tiempo % 100, 15))::INTEGER
                                                            AS dias_desde_1996,
    rem_tot,
    nullif(tramo_empleo, 0)                                 AS tramo_empleo,
    tr_anio_ini,
    nullif(provi, 0)                                        AS provi,
    nullif(letra, 0)                                        AS letra,
    nullif(r32, 0)                                          AS r32,
    nullif(r34, 0)                                          AS r34,
    fnacim,
    nullif(sexo, 0)                                         AS sexo
FROM read_csv('{csv}', header = true, columns = {tipos})
ORDER BY id_trabajador, tiempo, id_relacion
"""


def sha256(ruta, bloque: int = 2**24) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        while b := f.read(bloque):
            h.update(b)
    return h.hexdigest()


def ingerir(forzar: bool = False) -> None:
    if PANEL_BASE.exists() and not forzar:
        print(f"ya existe {PANEL_BASE} (usar --forzar para rehacerlo)")
        return
    PROCESADOS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    con = conectar()
    tmp = PANEL_BASE.with_suffix(".tmp")
    con.execute(
        f"COPY ({CONSULTA.format(csv=CSV_MLER, tipos=TIPOS_CSV)}) TO '{tmp}' "
        "(FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 1000000)"
    )
    tmp.rename(PANEL_BASE)
    n, n_trab = con.execute(f"SELECT count(*), count(DISTINCT id_trabajador) FROM '{PANEL_BASE}'").fetchone()
    print(f"panel_base: {n:,} filas, {n_trab:,} trabajadores, {time.time() - t0:.0f} s")

    meta = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "csv": str(CSV_MLER.name),
        "csv_bytes": CSV_MLER.stat().st_size,
        "csv_sha256": sha256(CSV_MLER),
        "filas": n,
        "trabajadores": n_trab,
    }
    META_INGESTA.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"metadatos en {META_INGESTA}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="MLER.csv → panel_base.parquet")
    ap.add_argument("--forzar", action="store_true")
    ingerir(ap.parse_args().forzar)
