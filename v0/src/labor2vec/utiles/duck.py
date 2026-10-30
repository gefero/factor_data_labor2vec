"""Conexión a DuckDB con límites adecuados a la máquina."""

import os

import duckdb

from labor2vec.utiles.rutas import PROCESADOS


def conectar(memoria: str | None = None, hilos: int | None = None) -> duckdb.DuckDBPyConnection:
    """Conexión en memoria que derrama a disco en data/processed/tmp_duckdb si hace falta.

    Por defecto usa un cuarto de la RAM y hasta 4 hilos: el contenedor comparte la máquina con VS Code
    y otros procesos, y con la mitad de la RAM y 8 hilos la máquina se quedó sin memoria y se tildó
    (2026-10-15, en el paso de salario relativo). memory_limit no es un tope duro: cada hilo suma
    buffers propios, por eso conviene dejar margen. Se ajusta con LABOR2VEC_DUCKDB_MEMORIA
    (p. ej. "8GB") y LABOR2VEC_DUCKDB_HILOS.
    """
    memoria = memoria or os.environ.get("LABOR2VEC_DUCKDB_MEMORIA")
    if memoria is None:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        memoria = f"{max(1, ram // 2**32)}GB"  # un cuarto, en GB
    hilos = hilos or int(os.environ.get("LABOR2VEC_DUCKDB_HILOS", min(4, os.cpu_count() or 4)))
    tmp = PROCESADOS / "tmp_duckdb"
    tmp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memoria}'; SET threads={hilos}; SET temp_directory='{tmp}'")
    con.execute("SET preserve_insertion_order=false")
    return con
