"""De los eventos de un trabajador al documento que consume el backbone.

Un documento son cuatro arreglos de igual largo: token, posición absoluta (días desde 1996-01-01), edad
(años) y segmento. Va primero el background (`[CLS] SEXO NAC [SEP]`, `N_FONDO` posiciones) y después los
eventos en orden cronológico, cada uno con un `[SEP]` al final.

La edad entra al modelo como entero (`pretrain.py` hace `input_ids.long()`), por eso se calcula en años cumplidos
aproximados (no hay mes de nacimiento).
"""

from dataclasses import dataclass

import numpy as np

from labor2vec.tokens.vocabulario import CLS, MASK, N_ESPECIALES, SEP, UNK, Vocabulario

N_FONDO = 4  # [CLS] SEXO NAC [SEP]
CICLO_SEGMENTOS = (2, 3, 1)  # los eventos recorren este ciclo (como `segment_pattern` del repo completo de life2vec)
# el background va siempre en segmento 1
SEG_FONDO = 1

# Orden de SOP
SOP_NORMAL, SOP_INVERTIDO, SOP_MEZCLADO = 0, 1, 2

# días desde 1996-01-01 al día 15 de cada mes (mes_idx 0 = 1996-01); cubre de sobra el panel (312 meses)
_DIAS_MES = (
    (np.datetime64("1996-01", "M") + np.arange(600).astype("timedelta64[M]")).astype("datetime64[D]") + np.timedelta64(14, "D") - np.datetime64("1996-01-01")
).astype(np.int32)


def dias_de_mes(mes_idx: np.ndarray) -> np.ndarray:
    return _DIAS_MES[mes_idx]


@dataclass
class Opciones:
    max_length: int = 2560
    fin_ventana: int | None = None  # último mes_idx incluido (239 = 2015-12); None = toda la historia
    eventos_compactos: bool = False  # variante B: sólo los eventos `nueva` llevan rama, provincia y antigüedad
    # "todos" (variante A) | "sin_ultimo_valor": cada evento es REM TAM [SEP], sin rama, provincia ni antigüedad (atributos de
    # «último valor» de la MLER, que pueden traer información posterior al corte; plans/PLAN_fase_8.md)
    atributos: str = "todos"
    # control de la fase 8: None | "tiempo" (se permutan fechas y edades entre los eventos de la persona) | "orden" (se
    # permuta el orden de los eventos en la secuencia, con su fecha). Necesita `rng`; con el de val/test es determinista.
    control_temporal: str | None = None
    # augmentación (probabilidades por documento, como en el config de light)
    p_remuestreo: float = 0.4
    rango_remuestreo: tuple[float, float] = (0.5, 0.75)  # fracción de eventos que se conserva
    p_ruido_tiempo: float = 0.1
    dias_ruido: int = 15
    p_ocultar_fondo: float = 0.01
    p_quitar_tokens: float = 0.01
    frac_quitar_tokens: float = 0.1
    # SOP
    p_sop_invertido: float = 0.05
    p_sop_mezclado: float = 0.05
    # MLM
    mask_ratio: float = 0.30


def opciones_de(d: dict, **extra) -> Opciones:
    """`Opciones` a partir de la sección `datos` de un YAML (max_length, fin_ventana y, si están, eventos_compactos,
    atributos y control_temporal); `extra` pisa campos sueltos (p. ej. `mask_ratio`)."""
    return Opciones(max_length=d["max_length"], fin_ventana=d["fin_ventana"],
                    eventos_compactos=d.get("eventos_compactos", False), atributos=d.get("atributos", "todos"),
                    control_temporal=d.get("control_temporal"), **extra)


@dataclass
class Eventos:
    """Eventos de un trabajador, ya ordenados por tiempo (columnas del corpus; -1 = dato faltante)."""

    mes_idx: np.ndarray
    pct_rem: np.ndarray
    tramo: np.ndarray
    r34: np.ndarray
    provi: np.ndarray
    antig: np.ndarray
    nueva: np.ndarray

    def __len__(self):
        return len(self.mes_idx)

    def tomar(self, idx) -> "Eventos":
        return Eventos(*(getattr(self, f)[idx] for f in self.__dataclass_fields__))


@dataclass
class Documento:
    tokens: np.ndarray
    abspos: np.ndarray
    edad: np.ndarray
    segmento: np.ndarray
    sop: int = SOP_NORMAL

    def __len__(self):
        return len(self.tokens)

    def como_matriz(self) -> np.ndarray:
        """(4, largo): token, posición absoluta, edad, segmento."""
        return np.stack([self.tokens, self.abspos, self.edad, self.segmento]).astype(np.int64)


def bloques_de_eventos(ev: Eventos, tablas: dict, compactos: bool,
                       atributos: str = "todos") -> tuple[np.ndarray, np.ndarray]:
    """Tokens de cada evento, concatenados, y largo de cada evento.

    Variante A: REM TAM RAMA PROV ANT [SEP]. Variante B: REM TAM [SEP] salvo en los eventos `nueva`, que
    llevan los seis tokens. Con `atributos="sin_ultimo_valor"`: REM TAM [SEP] en todos los eventos.
    """
    if atributos not in ("todos", "sin_ultimo_valor"):
        raise ValueError(f"atributos desconocido: {atributos!r}")
    if atributos == "sin_ultimo_valor" and compactos:
        raise ValueError("atributos='sin_ultimo_valor' no se combina con eventos_compactos")
    n = len(ev)
    m = np.empty((n, 6), dtype=np.int32)
    m[:, 0] = tablas["rem"][ev.pct_rem + 1]
    m[:, 1] = tablas["tam"][ev.tramo + 1]
    m[:, 2] = _buscar(tablas["rama"], ev.r34)
    m[:, 3] = _buscar(tablas["prov"], ev.provi)
    m[:, 4] = tablas["ant"][ev.antig + 1]
    m[:, 5] = SEP
    if atributos == "sin_ultimo_valor":
        return m[:, [0, 1, 5]].ravel(), np.full(n, 3, dtype=np.int32)
    if not compactos:
        return m.ravel(), np.full(n, 6, dtype=np.int32)
    completo = ev.nueva.astype(bool)
    # en un evento corto el [SEP] pasa a la columna 2; se descartan las columnas 2-4 y el sobrante
    m_corto = m.copy()
    m_corto[:, 2] = SEP
    mantener = np.zeros((n, 6), dtype=bool)
    mantener[:, :2] = True
    mantener[completo, :] = True
    mantener[~completo, 2] = True
    plano = np.where(completo[:, None], m, m_corto)[mantener]
    return plano, np.where(completo, 6, 3).astype(np.int32)


def _buscar(lut: np.ndarray, codigos: np.ndarray) -> np.ndarray:
    """Como lut[codigos + 1], pero los códigos fuera de la tabla pasan a [UNK]."""
    idx = codigos.astype(np.int64) + 1
    fuera = idx >= len(lut)
    return np.where(fuera, UNK, lut[np.minimum(idx, len(lut) - 1)])


def edad_en_anios(mes_idx: np.ndarray, fnacim: int) -> np.ndarray:
    anio = 1996 + mes_idx // 12
    mes = mes_idx % 12 + 1
    return np.floor(anio + (mes - 0.5) / 12 - (fnacim + 0.5)).astype(np.int32)


def construir_documento(ev: Eventos, sexo: int, fnacim: int, vocab: Vocabulario, tablas: dict, op: Opciones,
                        rng: np.random.Generator | None = None, augmentar: bool = False,
                        con_sop: bool = False) -> Documento:
    """Arma el documento de un trabajador. Con `augmentar` aplica la augmentación y con `con_sop`, la
    perturbación de orden de la tarea SOP; ambas necesitan `rng`."""
    if op.fin_ventana is not None:
        ev = ev.tomar(ev.mes_idx <= op.fin_ventana)
    dias = dias_de_mes(ev.mes_idx).astype(np.int32)
    edad = edad_en_anios(ev.mes_idx, fnacim)

    ocultar_fondo = quitar = False
    if augmentar:
        if len(ev) > 1 and rng.random() < op.p_remuestreo:
            frac = rng.uniform(*op.rango_remuestreo)
            conservar = np.sort(rng.choice(len(ev), size=max(1, int(round(frac * len(ev)))), replace=False))
            ev, dias, edad = ev.tomar(conservar), dias[conservar], edad[conservar]
        if len(ev) and rng.random() < op.p_ruido_tiempo:
            dias = np.maximum(0, dias + rng.integers(-op.dias_ruido, op.dias_ruido + 1, size=len(ev))).astype(np.int32)
        ocultar_fondo = rng.random() < op.p_ocultar_fondo
        quitar = rng.random() < op.p_quitar_tokens

    if op.control_temporal and len(ev) > 1:
        if rng is None:
            raise ValueError("control_temporal necesita rng")
        p = rng.permutation(len(ev))
        if op.control_temporal == "tiempo":  # mismos eventos en el mismo orden, sin saber cuándo pasó cada uno
            dias, edad = dias[p], edad[p]
        elif op.control_temporal == "orden":  # cada evento conserva su fecha pero cambia de lugar en la secuencia
            ev, dias, edad = ev.tomar(p), dias[p], edad[p]
        else:
            raise ValueError(f"control_temporal desconocido: {op.control_temporal!r}")

    plano, largos = bloques_de_eventos(ev, tablas, op.eventos_compactos, op.atributos)
    # truncado: se conservan los eventos más recientes que entran junto al background
    cupo = op.max_length - N_FONDO
    acum = np.cumsum(largos[::-1])[::-1]  # tokens de este evento y de todos los posteriores
    primero = int((acum > cupo).sum())
    if primero:
        plano = plano[largos[:primero].sum():]
        largos, dias, edad = largos[primero:], dias[primero:], edad[primero:]

    ids_sexo, ids_nac = vocab.fondo(sexo, fnacim)
    if ocultar_fondo:
        ids_sexo = ids_nac = UNK
    fondo = np.array([CLS, ids_sexo, ids_nac, SEP], dtype=np.int32)

    n_ev = len(largos)
    por_token = np.repeat(np.arange(n_ev), largos)
    ciclo = np.array(CICLO_SEGMENTOS, dtype=np.int32)
    tokens = np.concatenate([fondo, plano])
    abspos = np.concatenate([np.zeros(N_FONDO, np.int32), dias[por_token]])
    edades = np.concatenate([np.zeros(N_FONDO, np.int32), edad[por_token]])
    segmento = np.concatenate([np.full(N_FONDO, SEG_FONDO, np.int32), ciclo[por_token % len(ciclo)]])

    sop = SOP_NORMAL
    if con_sop and n_ev >= 2:
        u = rng.random()
        if u < op.p_sop_invertido:
            sop = SOP_INVERTIDO
        elif u < op.p_sop_invertido + op.p_sop_mezclado:
            sop = SOP_MEZCLADO
        if sop != SOP_NORMAL:
            orden = np.arange(n_ev)[::-1] if sop == SOP_INVERTIDO else _permutacion(rng, n_ev)
            inicio = np.concatenate([[0], np.cumsum(largos)])
            # los bloques cambian de lugar; la posición absoluta, la edad y el segmento quedan en su sitio
            tokens = np.concatenate([fondo] + [plano[inicio[i]:inicio[i + 1]] for i in orden])
    if quitar and len(tokens) > N_FONDO:
        cand = np.flatnonzero(tokens[N_FONDO:] != SEP) + N_FONDO
        k = int(op.frac_quitar_tokens * len(cand))
        if k:
            borrar = rng.choice(cand, size=k, replace=False)
            keep = np.ones(len(tokens), bool)
            keep[borrar] = False
            tokens, abspos, edades, segmento = tokens[keep], abspos[keep], edades[keep], segmento[keep]
    return Documento(tokens, abspos, edades, segmento, sop)


def _permutacion(rng: np.random.Generator, n: int) -> np.ndarray:
    """Permutación distinta de la identidad (n >= 2)."""
    while True:
        p = rng.permutation(n)
        if (p != np.arange(n)).any():
            return p


def enmascarar(doc_tokens: np.ndarray, vocab_size: int, mask_ratio: float, rng: np.random.Generator,
               n_mask: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MLM: devuelve (tokens con máscara, posiciones, objetivos). 80 % [MASK], 10 % sin cambio, 10 % aleatorio.

    No se enmascaran los especiales ([CLS], [SEP], [UNK], ...), que ocupan los ids < N_ESPECIALES.
    """
    cand = np.flatnonzero(doc_tokens >= N_ESPECIALES)
    k = min(int(np.floor(mask_ratio * len(doc_tokens))) if n_mask is None else n_mask, len(cand))
    pos = np.sort(rng.choice(cand, size=k, replace=False)) if k else np.zeros(0, dtype=np.int64)
    objetivo = doc_tokens[pos].copy()
    tok = doc_tokens.copy()
    u = rng.random(k)
    tok[pos[u < 0.8]] = MASK
    azar = (u >= 0.8) & (u < 0.9)
    tok[pos[azar]] = rng.integers(N_ESPECIALES, vocab_size, size=int(azar.sum()))
    return tok, pos, objetivo
