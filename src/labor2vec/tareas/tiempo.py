"""Variables de tiempo por trabajador para las líneas de base tabulares (fase 8b).

Las líneas de base de T1-T4 sólo ven proporciones de tokens: no saben cuándo pasó cada cosa. Estas variables resumen la
dinámica de la historia hasta 2015-12 (`mes_idx` 239) con la que el transformer sí cuenta: antigüedad, tiempo desde el
último cambio, cantidad de cambios recientes, huecos de registro y tendencia salarial. Sólo se usan eventos con
`mes_idx <= fin_ventana`.

La relación principal se define como en `derivadas.py`: la de más meses registrados en 2015 (desempate por mayor
remuneración promedio, luego por id). Si no tiene eventos en 2015 se toma la del último evento.

Hay dos conjuntos. `completo` tiene las 20 variables de `NOMBRES`. `sin_ultimo_valor` quita la antigüedad real y las
de cambios de rama (`RAMA` y `ANTIG`), para ser comparable con los modelos sin rama, provincia ni antigüedad.
"""

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

VENTANAS = (12, 24, 60)
NOMBRES = (["antiguedad"]
           + [f"meses_desde_cambio_{k}" for k in ("empresa", "tamano", "rama")]
           + [f"cambios_{k}_{w}" for k in ("empresa", "tamano", "rama") for w in VENTANAS]
           + ["huecos_24", "huecos_60", "meses_desde_ultimo_registro", "dif_salario_12", "dif_salario_36",
              "hay_dif_12", "hay_dif_36"])
ANTIG = {"antiguedad"}
RAMA = {n for n in NOMBRES if "rama" in n}
# se aplica log1p a lo que son meses y conteos (no a las diferencias de percentil ni a los indicadores)
SIN_LOG = {"dif_salario_12", "dif_salario_36", "hay_dif_12", "hay_dif_36"}
COLUMNAS = ["id_trabajador", "mes_idx", "pct_rem", "tramo", "r34", "id_relacion", "nueva"]
INICIO_2015 = 228


def indices(conjunto: str) -> list[int]:
    """Posiciones de `NOMBRES` que forman el conjunto (`completo` o `sin_ultimo_valor`)."""
    if conjunto == "completo":
        quitar = set()
    elif conjunto == "sin_ultimo_valor":
        quitar = ANTIG | RAMA
    else:
        raise ValueError(f"conjunto de variables de tiempo desconocido: {conjunto!r}")
    return [k for k, n in enumerate(NOMBRES) if n not in quitar]


def _relacion_principal(mes, rem, rel) -> int:
    en_2015 = mes >= INICIO_2015
    if not en_2015.any():
        return int(rel[-1])
    r, m, p = rel[en_2015], mes[en_2015], rem[en_2015].astype(float)
    candidatas = np.unique(r)
    meses = np.array([len(np.unique(m[r == c])) for c in candidatas])
    prom = np.array([p[r == c].mean() for c in candidatas])
    orden = np.lexsort((candidatas, -prom, -meses))  # más meses, mayor remuneración, menor id
    return int(candidatas[orden[0]])


def _cambios(meses_cambio: np.ndarray, corte: int, desde: int) -> list[float]:
    """Meses desde el último cambio (sin cambio: meses desde `desde`) y cantidad de cambios por ventana."""
    ult = corte - int(meses_cambio.max()) if len(meses_cambio) else corte - desde
    return [float(ult)] + [float((meses_cambio > corte - w).sum()) for w in VENTANAS]


def _max_por_mes(mes: np.ndarray, val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(meses únicos, máximo de `val` en cada mes) para `mes` ordenado."""
    ini = np.flatnonzero(np.r_[True, np.diff(mes) != 0])
    return mes[ini], np.maximum.reduceat(val, ini)


def _valor_en_o_antes(serie_mes: np.ndarray, serie_val: np.ndarray, mes: int):
    k = np.searchsorted(serie_mes, mes, side="right") - 1
    return None if k < 0 else float(serie_val[k])


def variables(mes, rem, tramo, rama, rel, nueva, fin_ventana: int = 239) -> np.ndarray:
    """Vector de `NOMBRES` de un trabajador a partir de sus eventos (arreglos del mismo largo, ordenados por mes)."""
    sel = mes <= fin_ventana
    mes, rem, tramo, rama, rel, nueva = (a[sel] for a in (mes, rem, tramo, rama, rel, nueva))
    out = np.zeros(len(NOMBRES))
    if len(mes) == 0:
        return out
    corte, primero = fin_ventana, int(mes.min())
    meses_unicos = np.unique(mes)  # `mes` ya viene ordenado

    p = rel == _relacion_principal(mes, rem, rel)
    mp, tp = mes[p], tramo[p]
    out[0] = corte - int(mp.min()) + 1  # antigüedad en la relación principal, en meses

    # inicios de relación posteriores al primer mes de la historia (el primero no es un cambio)
    inicios = np.unique(mes[nueva & (mes > primero)])
    # cambios de tamaño dentro de la relación principal (un valor por mes: el máximo del mes)
    um, t_mes = _max_por_mes(mp, tp)
    camb_tam = um[1:][np.diff(t_mes) != 0]
    # la rama es de «último valor»: no cambia dentro de una relación, así que un cambio de rama es que la relación que
    # empieza tenga otra rama que la anterior
    mes_ini, rama_ini = mes[nueva], rama[nueva]
    camb_rama = mes_ini[1:][np.diff(rama_ini) != 0] if len(mes_ini) > 1 else mes_ini[:0]
    desde = int(mp.min())
    for k, c in enumerate((inicios, camb_tam, camb_rama)):
        v = _cambios(c, corte, primero if k == 0 else desde)
        out[1 + k] = v[0]
        out[4 + 3 * k: 7 + 3 * k] = v[1:]

    for j, w in enumerate((24, 60)):
        ini = max(corte - w + 1, primero)
        out[13 + j] = (corte - ini + 1) - int(((meses_unicos >= ini) & (meses_unicos <= corte)).sum())
    out[15] = corte - int(mes.max())

    salario = _max_por_mes(mes, rem)[1].astype(float)
    actual = salario[-1]
    for j, w in enumerate((12, 36)):
        antes = _valor_en_o_antes(meses_unicos, salario, corte - w)
        if antes is not None:
            out[16 + j], out[18 + j] = actual - antes, 1.0
    return out


def calcular(directorio, ids: np.ndarray | None = None, fin_ventana: int = 239) -> tuple[np.ndarray, np.ndarray]:
    """(id_trabajador, matriz (n, len(NOMBRES))) de todos los trabajadores del corpus en `directorio` (o sólo `ids`)."""
    archivos = sorted(Path(directorio).glob("*.parquet"))
    tabla = pq.ParquetDataset(archivos).read(columns=COLUMNAS).combine_chunks()
    ident = tabla["id_trabajador"].to_numpy()
    lista = tabla["mes_idx"].chunk(0)
    off = lista.offsets.to_numpy().astype(np.int64)
    col = {c: tabla[c].chunk(0).values.to_numpy(zero_copy_only=False) for c in COLUMNAS[1:]}
    quiero = None if ids is None else {int(i) for i in ids}
    filas, vistos = [], []
    for i in range(len(ident)):
        if quiero is not None and int(ident[i]) not in quiero:
            continue
        a, b = off[i], off[i + 1]
        filas.append(variables(col["mes_idx"][a:b], col["pct_rem"][a:b], col["tramo"][a:b], col["r34"][a:b],
                               col["id_relacion"][a:b], col["nueva"][a:b], fin_ventana))
        vistos.append(int(ident[i]))
    return np.array(vistos, dtype=np.int64), np.array(filas).reshape(len(filas), len(NOMBRES))


def transformar(x: np.ndarray, media: np.ndarray | None = None, desvio: np.ndarray | None = None):
    """log1p de meses y conteos y estandarización con la media y el desvío de train. Devuelve (x, media, desvío)."""
    x = x.copy()
    for k, n in enumerate(NOMBRES):
        if n not in SIN_LOG:
            x[:, k] = np.log1p(np.maximum(x[:, k], 0))
    if media is None:
        media, desvio = x.mean(0), np.maximum(x.std(0), 1e-6)
    return ((x - media) / desvio).astype(np.float32), media, desvio
