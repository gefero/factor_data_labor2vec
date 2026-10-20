"""Rutas del proyecto. Todas relativas a la raíz del repo (o a LABOR2VEC_RAIZ, si está definida)."""

import os
from pathlib import Path

RAIZ = Path(os.environ.get("LABOR2VEC_RAIZ", Path(__file__).resolve().parents[3]))

DATA = RAIZ / "data"
DOCS = DATA / "docs"
PROCESADOS = DATA / "processed"
INFORMES = RAIZ / "informes"

CSV_MLER = DATA / "MLER.csv"
XLSX_ANEXO = DOCS / "Anexo_estadistico_y_descriptores.xlsx"

PANEL_BASE = PROCESADOS / "panel_base.parquet"
PANEL = PROCESADOS / "panel.parquet"
META_INGESTA = PROCESADOS / "ingesta.json"
DESCRIPTORES = PROCESADOS / "descriptores"
TRABAJADORES = PROCESADOS / "trabajadores.parquet"
RELACIONES = PROCESADOS / "relaciones.parquet"
SPLITS = PROCESADOS / "splits.parquet"
CORTES_SALARIO = PROCESADOS / "cortes_salario_mensual.parquet"

CORPUS = PROCESADOS / "corpus"
VOCABULARIO = CORPUS / "vocabulario.parquet"
RAMAS = CORPUS / "ramas.parquet"

TAREAS = PROCESADOS / "tareas"
T1 = TAREAS / "t1.parquet"
