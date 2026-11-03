"""De los eventos de un trabajador al documento que consume el backbone.

Un documento son cuatro arreglos de igual largo: token, posición absoluta (días desde 1996-01-01), edad
(años) y segmento. Va primero el background (`[CLS] SEXO NAC [SEP]`, `N_FONDO` posiciones) y después los
eventos en orden cronológico, cada uno con un `[SEP]` al final.

Con `Opciones.grilla` (v1, plans/PLAN_v1.md) cada mes sin registro, desde el primer mes observado hasta el fin de la
ventana, es un evento `SINREG_x [SEP]`: intermitencia (hay un registro posterior en la ventana) o salida (no lo hay).

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

# Tipo de evento (`Eventos.tipo`): empleo o mes sin registro
TIPO_EMPLEO, TIPO_INT, TIPO_SAL = 0, 1, 2
FIN_PANEL = 311  # último mes_idx de la MLER (2021-12), fin de la grilla cuando no hay ventana
SINREG_MODOS = ("dos", "uno")  # "uno": un solo token para los dos tipos (control; se usa el id de SINREG_INT)
SINREG_BLOQUE = 2  # tokens de un mes sin registro: SINREG_x [SEP]

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
    eventos_compactos: bool = False  # variante B: sólo los eventos `nueva` llevan rama, provincia e inicio de la empresa
    # "todos" (variante A) | "sin_ultimo_valor": cada evento es REM TAM [SEP], sin rama, provincia ni inicio de la empresa (atributos de
    # «último valor» de la MLER, que pueden traer información posterior al corte; plans/PLAN_fase_8.md)
    atributos: str = "todos"
    # v1: un evento SINREG por cada mes sin registro entre el primer mes observado y el fin de la ventana
    grilla: bool = False
    sinreg: str = "dos"  # "dos" (SINREG_INT y SINREG_SAL) | "uno" (un solo token; control)
    # control de la fase 8: None | "tiempo" (se permutan fechas y edades entre los eventos de la persona) | "orden" (se
    # permuta el orden de los eventos en la secuencia, con su fecha). Necesita `rng`; con el de val/test es determinista.
    control_temporal: str | None = None
    # intervención de la fase 9 (B3), una de `INTERVENCIONES`: cambia una cosa de la historia antes de tokenizar
    intervencion: str | None = None
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
    atributos, control_temporal, intervencion, grilla y sinreg); `extra` pisa campos sueltos (p. ej. `mask_ratio`)."""
    return Opciones(max_length=d["max_length"], fin_ventana=d["fin_ventana"],
                    eventos_compactos=d.get("eventos_compactos", False), atributos=d.get("atributos", "todos"),
                    control_temporal=d.get("control_temporal"), intervencion=d.get("intervencion"),
                    grilla=d.get("grilla", False), sinreg=d.get("sinreg", "dos"), **extra)


@dataclass
class Eventos:
    """Eventos de un trabajador, ya ordenados por tiempo (columnas del corpus; -1 = dato faltante)."""

    mes_idx: np.ndarray
    pct_rem: np.ndarray
    tramo: np.ndarray
    r34: np.ndarray
    provi: np.ndarray
    empini: np.ndarray  # inicio de actividades de la empresa (tr_anio_ini; la columna del corpus se llama `antig`)
    nueva: np.ndarray
    tipo: np.ndarray | None = None  # TIPO_EMPLEO (0) | TIPO_INT (1) | TIPO_SAL (2); None = todos de empleo

    def __post_init__(self):
        if self.tipo is None:
            self.tipo = np.zeros(len(self.mes_idx), dtype=np.int8)

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


def bloques_de_eventos(ev: Eventos, tablas: dict, compactos: bool, atributos: str = "todos",
                       sinreg: str = "dos") -> tuple[np.ndarray, np.ndarray]:
    """Tokens de cada evento, concatenados, y largo de cada evento.

    Variante A: REM TAM RAMA PROV EMPINI [SEP]. Variante B: REM TAM [SEP] salvo en los eventos `nueva`, que
    llevan los seis tokens. Con `atributos="sin_ultimo_valor"`: REM TAM [SEP] en todos los eventos. Los eventos de
    mes sin registro (`ev.tipo > 0`) son `SINREG_INT [SEP]` o `SINREG_SAL [SEP]` (con `sinreg="uno"`, siempre el primero).
    """
    if sinreg not in SINREG_MODOS:
        raise ValueError(f"sinreg desconocido: {sinreg!r}")
    hueco = ev.tipo != TIPO_EMPLEO
    if not hueco.any():
        return _bloques_de_empleo(ev, tablas, compactos, atributos)
    plano_emp, largos_emp = _bloques_de_empleo(ev.tomar(~hueco), tablas, compactos, atributos)
    largos = np.full(len(ev), SINREG_BLOQUE, dtype=np.int32)
    largos[~hueco] = largos_emp
    token_hueco = np.repeat(hueco, largos)
    plano = np.empty(int(largos.sum()), dtype=np.int32)
    plano[~token_hueco] = plano_emp
    ids = tablas["sinreg"][np.zeros(int(hueco.sum()), dtype=np.int64) if sinreg == "uno" else ev.tipo[hueco] - 1]
    plano[token_hueco] = np.stack([ids, np.full(len(ids), SEP, dtype=np.int32)], axis=1).ravel()
    return plano, largos


def _bloques_de_empleo(ev: Eventos, tablas: dict, compactos: bool,
                       atributos: str = "todos") -> tuple[np.ndarray, np.ndarray]:
    """`bloques_de_eventos` para eventos que son todos de empleo."""
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
    m[:, 4] = tablas["empini"][ev.empini + 1]
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


INTERVENCIONES = ("sin_ultimo_anio", "sin_cambios_recientes", "salario_arriba", "salario_abajo", "atrasar_2_anios")
PUNTOS_SALARIO = 10  # puntos de percentil (un decil) que suben o bajan los salarios del último año


def intervenir(ev: Eventos, nombre: str, fin: int) -> Eventos:
    """Fase 9 (B3): variante de la historia de una persona; `fin` es el último mes de la ventana (239 = 2015-12).

    - `sin_ultimo_anio`: se borran los eventos de los últimos 12 meses.
    - `sin_cambios_recientes`: la última relación conserva la rama y el tamaño de la anterior (relaciones = eventos desde
      cada `nueva`; con simultáneas es una aproximación). Si hay una sola relación no cambia nada.
    - `salario_arriba` / `salario_abajo`: el percentil de los eventos del último año sube o baja un decil (entre 1 y 100;
      los meses con remuneración 0 no se tocan).
    - `atrasar_2_anios`: todas las fechas, y con ellas las edades, pasan 24 meses atrás; lo que queda antes del mes 0 se descarta.
    Devuelve copias: no modifica `ev`, que viene de la memoria del corpus.
    """
    if nombre not in INTERVENCIONES:
        raise ValueError(f"intervencion desconocida: {nombre!r}")
    if nombre == "sin_ultimo_anio":
        return ev.tomar(ev.mes_idx <= fin - 12)
    out = Eventos(*(getattr(ev, f).copy() for f in ev.__dataclass_fields__))
    if nombre == "atrasar_2_anios":
        out.mes_idx = (out.mes_idx.astype(np.int32) - 24).astype(ev.mes_idx.dtype)
        return out.tomar(out.mes_idx >= 0)
    if nombre in ("salario_arriba", "salario_abajo"):
        signo = 1 if nombre == "salario_arriba" else -1
        sel = (out.mes_idx > fin - 12) & (out.pct_rem > 0)
        nuevo = np.clip(out.pct_rem[sel].astype(np.int32) + signo * PUNTOS_SALARIO, 1, 100)
        out.pct_rem[sel] = nuevo.astype(out.pct_rem.dtype)
        return out
    inicios = np.flatnonzero(out.nueva.astype(bool))
    if len(inicios) >= 2:
        k = inicios[-1]
        out.r34[k:], out.tramo[k:] = out.r34[k - 1], out.tramo[k - 1]
    return out


def fin_de_la_historia(fin: int | None, intervencion: str | None) -> int:
    """Último mes de la grilla: el fin de la ventana (el del panel si no hay ventana), corrido si la intervención de la
    fase 9 cambia hasta dónde llega la historia (`sin_ultimo_anio` la acorta 12 meses; `atrasar_2_anios` la corre 24)."""
    fin = FIN_PANEL if fin is None else fin
    return fin - {"sin_ultimo_anio": 12, "atrasar_2_anios": 24}.get(intervencion, 0)


def con_grilla(ev: Eventos, fin: int) -> Eventos:
    """Agrega un evento por cada mes sin registro entre el primer mes de `ev` y `fin`, en su lugar cronológico.

    El tipo depende sólo de lo que hay hasta `fin`: intermitencia si hay algún registro posterior (`TIPO_INT`) y salida si
    el último registro quedó atrás (`TIPO_SAL`). Los meses anteriores al primer registro no se marcan (el modelo los lee
    por la edad y la fecha del primer evento). Los eventos de empleo conservan su orden."""
    if len(ev) == 0:
        return ev
    mes = ev.mes_idx.astype(np.int32)
    ultimo = int(mes.max())
    vacios = np.setdiff1d(np.arange(int(mes.min()), max(fin, ultimo) + 1), mes)
    if len(vacios) == 0:
        return ev
    n = len(vacios)

    def falta(ref):
        return np.full(n, -1, dtype=ref.dtype)

    huecos = Eventos(mes_idx=vacios.astype(ev.mes_idx.dtype), pct_rem=falta(ev.pct_rem), tramo=falta(ev.tramo),
                     r34=falta(ev.r34), provi=falta(ev.provi), empini=falta(ev.empini),
                     nueva=np.zeros(n, dtype=ev.nueva.dtype),
                     tipo=np.where(vacios > ultimo, TIPO_SAL, TIPO_INT).astype(np.int8))
    todos = Eventos(*(np.concatenate([getattr(ev, f), getattr(huecos, f)]) for f in ev.__dataclass_fields__))
    return todos.tomar(np.argsort(todos.mes_idx, kind="stable"))


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
    if op.intervencion:
        fin = op.fin_ventana if op.fin_ventana is not None else (int(ev.mes_idx.max()) if len(ev) else 0)
        ev = intervenir(ev, op.intervencion, fin)
    if op.grilla:
        ev = con_grilla(ev, fin_de_la_historia(op.fin_ventana, op.intervencion))
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

    plano, largos = bloques_de_eventos(ev, tablas, op.eventos_compactos, op.atributos, op.sinreg)
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
    if con_sop and _hay_orden_que_perturbar(largos):
        u = rng.random()
        if u < op.p_sop_invertido:
            sop = SOP_INVERTIDO
        elif u < op.p_sop_invertido + op.p_sop_mezclado:
            sop = SOP_MEZCLADO
        if sop != SOP_NORMAL:
            orden = _orden_sop(rng, largos, sop)
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


def _hay_orden_que_perturbar(largos: np.ndarray) -> bool:
    """El SOP necesita al menos dos bloques del mismo largo que intercambiar (con un solo largo, como en v0: dos eventos)."""
    return len(largos) >= 2 and np.bincount(largos).max() >= 2


def _orden_sop(rng: np.random.Generator, largos: np.ndarray, sop: int) -> np.ndarray:
    """Orden de los bloques tras la perturbación: `orden[j]` es el bloque que queda en el lugar j.

    Sólo se intercambian bloques del mismo largo (meses con empleo entre sí, meses sin registro entre sí): así cada
    [SEP], fecha, edad y segmento queda en su lugar y la perturbación no se delata por la posición de los [SEP] (con la
    grilla de v1 los bloques miden 6 o 2 y una permutación libre los desalineaba en el 86 % de los documentos). Con un
    solo largo es la inversión o la permutación de todos los bloques, igual que en v0."""
    orden = np.arange(len(largos))
    for largo in np.unique(largos):
        lugares = np.flatnonzero(largos == largo)
        if len(lugares) < 2:
            continue
        orden[lugares] = lugares[::-1] if sop == SOP_INVERTIDO else lugares[_permutacion(rng, len(lugares))]
    return orden


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
