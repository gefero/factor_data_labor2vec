"""Rutas del proyecto. Todas relativas a la raíz de v1.1 (o a LABOR2VEC_RAIZ, si está definida).

Los datos crudos (MLER.csv y la documentación) son compartidos y viven en `data/` de la raíz del repo, un nivel
arriba de v1_1. Los derivados (`data/processed/`: panel, corpus, cohortes) son los de v0: `v1_1/data/processed` es un
enlace simbólico a `../../v0/data/processed`. Lo que genera v1.1 va a `v1_1/data/` y `v1_1/runs/`."""

import os
from pathlib import Path

RAIZ = Path(os.environ.get("LABOR2VEC_RAIZ", Path(__file__).resolve().parents[3]))

DATA_CRUDOS = Path(os.environ.get("LABOR2VEC_DATA", RAIZ.parent / "data"))
DATA = RAIZ / "data"
DOCS = DATA_CRUDOS / "docs"
PROCESADOS = DATA / "processed"
INFORMES = RAIZ / "informes"

CSV_MLER = DATA_CRUDOS / "MLER.csv"
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
