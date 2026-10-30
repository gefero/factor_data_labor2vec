"""Fase 9: análisis de secuencias clásico (optimal matching) para comparar con el espacio del modelo (plans/PLAN_fase_9.md).

Cada persona es una secuencia de estados mensuales: 0 = sin registro y, si hay empleo, 1 + 3·salario + tamaño con salario
en terciles de `pct_rem_mes` (cortes fijos 33 y 67) y tamaño 0 = chica (tramo 3-4), 1 = mediana (tramo 2), 2 = grande
(tramo 1). La relación del mes es la de mayor remuneración. Las distancias de optimal matching usan costos de
sustitución TRATE (a partir de las transiciones mensuales observadas) e inserción y borrado de costo fijo, como el uso
estándar de TraMineR.
"""

from pathlib import Path

import numpy as np
import pyarrow as pa
from numba import njit, prange

from labor2vec.tareas.t1 import CORTE
from labor2vec.utiles.duck import conectar
from labor2vec.utiles.rutas import PANEL

N_ESTADOS = 10
CORTES_SALARIO = (33.0, 67.0)
MESES = 120  # 2006-01 a 2015-12


def sql_estados(panel: Path, ids_tabla: str, mes0: int, mes1: int) -> str:
    """Una fila por (persona, mes) con empleo: el estado de la relación de mayor remuneración de ese mes."""
    s1, s2 = CORTES_SALARIO
    return f"""
SELECT id_trabajador, mes_idx,
       1 + 3 * (CASE WHEN pct_rem_mes < {s1} THEN 0 WHEN pct_rem_mes < {s2} THEN 1 ELSE 2 END)
         + (CASE WHEN tramo_empleo = 1 THEN 2 WHEN tramo_empleo = 2 THEN 1 ELSE 0 END) AS estado
FROM (
    SELECT p.*, row_number() OVER (PARTITION BY p.id_trabajador, p.mes_idx
                                   ORDER BY p.rem_tot DESC NULLS LAST, p.id_relacion) AS rk
    FROM '{panel}' p JOIN {ids_tabla} USING (id_trabajador)
    WHERE p.mes_idx BETWEEN {mes0} AND {mes1}) WHERE rk = 1
"""


def estados_mensuales(ids: np.ndarray, fin: int = CORTE, meses: int = MESES, panel: Path = PANEL) -> np.ndarray:
    """(n, meses) uint8 con el estado de cada persona de `ids` (en ese orden) en los `meses` que terminan en `fin`."""
    con = conectar()
    con.register("ids_seq", pa.table({"id_trabajador": np.asarray(ids, dtype=np.int64)}))
    mes0 = fin - meses + 1
    t = con.execute(sql_estados(panel, "ids_seq", mes0, fin)).pl()
    fila = {int(i): k for k, i in enumerate(ids)}
    seqs = np.zeros((len(ids), meses), dtype=np.uint8)
    f = np.array([fila[int(i)] for i in t["id_trabajador"]], dtype=np.int64)
    seqs[f, t["mes_idx"].to_numpy().astype(np.int64) - mes0] = t["estado"].to_numpy().astype(np.uint8)
    return seqs


def costos_trate(seqs: np.ndarray, n_estados: int = N_ESTADOS) -> np.ndarray:
    """Costos de sustitución TRATE: 2 − P(i→j) − P(j→i) (diagonal 0), con P la proporción de transiciones mensuales."""
    cuentas = np.zeros((n_estados, n_estados))
    np.add.at(cuentas, (seqs[:, :-1].ravel(), seqs[:, 1:].ravel()), 1)
    p = cuentas / np.maximum(cuentas.sum(1, keepdims=True), 1)
    c = 2.0 - p - p.T
    np.fill_diagonal(c, 0.0)
    return c


@njit(cache=True)
def _distancia(a, b, sub, indel):
    n, m = len(a), len(b)
    previa = np.arange(m + 1) * indel
    actual = np.empty(m + 1)
    for i in range(1, n + 1):
        actual[0] = i * indel
        for j in range(1, m + 1):
            v = previa[j - 1] + sub[a[i - 1], b[j - 1]]
            v = min(v, previa[j] + indel, actual[j - 1] + indel)
            actual[j] = v
        previa, actual = actual, previa
    return previa[m]


@njit(parallel=True, cache=True)
def _om(seqs, sub, indel):
    n = len(seqs)
    out = np.empty(n * (n - 1) // 2, dtype=np.float32)
    for i in prange(n - 1):
        base = i * n - i * (i + 1) // 2  # posición de (i, i+1) en el formato condensado de scipy
        for j in range(i + 1, n):
            out[base + j - i - 1] = _distancia(seqs[i], seqs[j], sub, indel)
    return out


def distancias_om(seqs: np.ndarray, sub: np.ndarray, indel: float = 1.0) -> np.ndarray:
    """Distancias de optimal matching entre todos los pares, en formato condensado (como `scipy.spatial.distance.pdist`)."""
    return _om(np.ascontiguousarray(seqs), np.ascontiguousarray(sub, dtype=np.float64), float(indel))
