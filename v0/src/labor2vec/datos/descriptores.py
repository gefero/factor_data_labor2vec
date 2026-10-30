"""Descriptores de códigos de la MLER → data/processed/descriptores/*.parquet.

Lee las hojas DESC_* del Anexo estadístico del MTEySS y agrega una tabla propia de regiones
para las provincias (regiones estadísticas del INDEC usadas en la EPH). Verifica que todos los
códigos presentes en el panel tengan descripción y la jerarquía r34 → r32 → letra.

    python -m labor2vec.datos.descriptores
"""

import openpyxl
import polars as pl

from labor2vec.utiles.duck import conectar
from labor2vec.utiles.rutas import DESCRIPTORES, PANEL_BASE, XLSX_ANEXO

# hoja → (nombre de la tabla, columna de código)
HOJAS = {
    "DESC_SEXO": ("sexo", "sexo"),
    "DESC_PROVI": ("provi", "provi"),
    "DESC_LETRA": ("letra", "letra"),
    "DESC_R32": ("r32", "r32"),
    "DESC_R34": ("r34", "r34"),
    "DESC_TRAMO EMPLEO": ("tramo_empleo", "tramo_empleo"),
    "DESC_TR_ANIO_INI": ("tr_anio_ini", "tr_anio_ini"),
}

# Regiones estadísticas del INDEC (EPH). El GBA comprende CABA y los partidos del conurbano.
REGIONES = {
    1: "GBA", 2: "GBA",
    6: "Pampeana", 14: "Pampeana", 30: "Pampeana", 42: "Pampeana", 82: "Pampeana",
    10: "NOA", 38: "NOA", 46: "NOA", 66: "NOA", 86: "NOA", 90: "NOA",
    18: "NEA", 22: "NEA", 34: "NEA", 54: "NEA",
    50: "Cuyo", 70: "Cuyo", 74: "Cuyo",
    26: "Patagonia", 58: "Patagonia", 62: "Patagonia", 78: "Patagonia", 94: "Patagonia",
}


def leer_hoja(ws, columna: str) -> pl.DataFrame:
    """Filas con código entero en la primera columna; la descripción es la última celda no vacía.
    En DESC_LETRA hay una columna extra con la letra CIIU (A, B, ...)."""
    filas = []
    for r in ws.iter_rows(values_only=True):
        if not r or not isinstance(r[0], (int, float)):
            continue
        celdas = [c for c in r if c is not None]
        fila = {columna: int(r[0]), "descripcion": str(celdas[-1]).strip()}
        if columna == "letra":
            fila["letra_ciiu"] = r[1]
        filas.append(fila)
    return pl.DataFrame(filas)


def construir() -> dict[str, pl.DataFrame]:
    wb = openpyxl.load_workbook(XLSX_ANEXO, read_only=True, data_only=True)
    tablas = {nombre: leer_hoja(wb[hoja], col) for hoja, (nombre, col) in HOJAS.items()}
    tablas["provi"] = tablas["provi"].with_columns(
        pl.col("provi").replace_strict(REGIONES, default=None).alias("region")
    )
    DESCRIPTORES.mkdir(parents=True, exist_ok=True)
    for nombre, df in tablas.items():
        df.write_parquet(DESCRIPTORES / f"{nombre}.parquet")
        print(f"{nombre}: {df.height} códigos")
    return tablas


def verificar(tablas: dict[str, pl.DataFrame]) -> None:
    con = conectar()
    problemas = []
    for nombre, df in tablas.items():
        codigos = set(df[nombre].to_list())
        en_panel = {c for (c,) in con.execute(
            f"SELECT DISTINCT {nombre} FROM '{PANEL_BASE}' WHERE {nombre} IS NOT NULL").fetchall()}
        faltan = sorted(en_panel - codigos)
        if faltan:
            problemas.append(f"{nombre}: códigos sin descripción {faltan}")
    malos = con.execute(
        f"SELECT count(*) FILTER (WHERE r34 // 100 <> r32) FROM '{PANEL_BASE}' WHERE r34 IS NOT NULL"
    ).fetchone()[0]
    if malos:
        problemas.append(f"jerarquía: {malos} filas con r32 != r34 // 100")
    letras_por_r32 = con.execute(f"""
        SELECT count(*) FROM (SELECT r32 FROM '{PANEL_BASE}' WHERE r32 IS NOT NULL
                              GROUP BY r32 HAVING count(DISTINCT letra) > 1)""").fetchone()[0]
    if letras_por_r32:
        problemas.append(f"jerarquía: {letras_por_r32} códigos r32 con más de una letra")
    sin_region = tablas["provi"].filter(pl.col("region").is_null() & (pl.col("provi") != 0))
    if sin_region.height:
        problemas.append(f"provincias sin región: {sin_region['provi'].to_list()}")
    if problemas:
        raise ValueError("descriptores inconsistentes:\n  " + "\n  ".join(problemas))
    print("verificación OK: todos los códigos del panel tienen descripción; jerarquía r34 → r32 → letra válida")


if __name__ == "__main__":
    verificar(construir())
