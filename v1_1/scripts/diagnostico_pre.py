"""Diagnósticos del pre-entrenamiento con casos pareados entre v0 y v1 (v1/informes/fase3_evaluacion.md, bloque A).

    PYTHONPATH=src ../v0/.venv/bin/python scripts/diagnostico_pre.py preparar [--cada 20] [--salida runs/diagnostico_pre]
    PYTHONPATH=src ../v0/.venv/bin/python scripts/diagnostico_pre.py inferir <corrida> {v0,v1} [--nombre v1_s0]
    PYTHONPATH=src ../v0/.venv/bin/python scripts/diagnostico_pre.py resumir v1_s0 v0_s0 v0_s1 v0_s2 [--remuestreos 2000]

Desde v1.1, `preparar --sop-v11` altera los documentos con grilla con el SOP de v1.1 (sin mover los SINREG); los casos
del MLM y los documentos de v0 no cambian, así que los resultados de v0 de `v1/runs/diagnostico_pre/` siguen valiendo.

`preparar` (sin modelo) arma, para uno de cada `--cada` trabajadores de validación, el documento de v0 (sin grilla) y el de
v1 (con grilla) y tapa en los dos **los mismos casos** (trabajador, evento de empleo, campo), elegidos en coordenadas de
evento con la tasa de v0 (piso de 0,3 × largo del documento de v0). Los `SINREG` no se tapan y no hay alteración del SOP.
Además arma un documento alterado por trabajador (invertido los de índice par, mezclado los impares) con el SOP de cada
versión y lo clasifica en «invisible» (idéntico al normal), «atajo» (un `SINREG` fuera de su lugar, imposible en un
documento normal) o «resto». Guarda por caso la categoría, la posición en la trayectoria y la predicción de la regla
«copiar al vecino» (mismo campo del evento de empleo visible más cercano, primero hacia atrás).

`inferir` pasa un modelo (`config.json` y `modelo_final.pt`; el código del modelo es el mismo en v0 y v1, y el tamaño
del vocabulario sale del checkpoint) por los documentos de su versión: NLL y acierto de cada caso y P(normal) del SOP.

`resumir` arma las tablas D1-D6 con intervalos del 95 % por remuestreo de trabajadores y las guarda en `resumen.json`.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from labor2vec.modelo.config import hparams
from labor2vec.modelo.pretrain import TransformerEncoder
from labor2vec.tokens.datos import DatasetPretrain, collate
from labor2vec.tokens.documento import (
    N_FONDO,
    SOP_NORMAL,
    TIPO_EMPLEO,
    TIPO_INT,
    Opciones,
    con_grilla,
    fin_de_la_historia,
)
from labor2vec.tokens.vocabulario import MASK, N_ESPECIALES, SEP, Vocabulario
from labor2vec.utiles.rutas import CORPUS

FIN = 239  # 2015-12
CAMPOS = ("salario", "tamaño", "rama", "provincia", "inicio_empresa")  # orden de los tokens de un evento de empleo
CATEGORIAS = CAMPOS + ("sexo", "nacimiento")
POSICIONES = ("primero", "continuidad", "nueva_sin_hueco", "tras_hueco", "fondo")
GRUPOS_SOP = ("invisible", "atajo", "resto")
VOCAB_V0 = 519  # los reemplazos al azar del 10 % salen de los tokens que existen en las dos versiones
SALIDA = Path("runs/diagnostico_pre")


# ---- preparar ------------------------------------------------------------------------------------------------------
def _opciones(grilla: bool, invertido: float = 0.0, mezclado: float = 0.0, sop_v11: bool = False) -> Opciones:
    return Opciones(max_length=2560, fin_ventana=FIN, grilla=grilla, p_sop_invertido=invertido, p_sop_mezclado=mezclado,
                    sop_sinreg_fijo=sop_v11 and grilla)


def _inicios(largos: np.ndarray, n_tokens: int) -> tuple[np.ndarray, int]:
    """Posición del primer token de cada evento y cantidad de eventos descartados por el truncado."""
    inicio = N_FONDO + np.concatenate([[0], np.cumsum(largos)[:-1]])
    descartados = int(np.flatnonzero(N_FONDO + largos.sum() - (inicio - N_FONDO) == n_tokens)[0])
    return inicio - (inicio[descartados] - N_FONDO), descartados


def _sinreg_fuera_de_lugar(tokens: np.ndarray, ids_int: int, ids_sal: int) -> bool:
    """Un `SINREG_SAL` antes del último empleo o un `SINREG_INT` después: nunca pasa en un documento normal."""
    t = tokens[N_FONDO:]
    emp = np.flatnonzero((t != SEP) & (t != ids_int) & (t != ids_sal))
    if len(emp) == 0:
        return False
    s, i = np.flatnonzero(t == ids_sal), np.flatnonzero(t == ids_int)
    return bool((len(s) and s.min() < emp[-1]) or (len(i) and i.max() > emp[-1]))


def _enmascarar_en(tokens: np.ndarray, pos: np.ndarray, u: np.ndarray, azar: np.ndarray) -> np.ndarray:
    """80 % [MASK], 10 % token al azar, 10 % sin cambio, con los sorteos dados (iguales en las dos versiones)."""
    tok = tokens.copy()
    tok[pos[u < 0.8]] = MASK
    sel = (u >= 0.8) & (u < 0.9)
    tok[pos[sel]] = azar[sel]
    return tok


def preparar(a) -> None:
    vocab = Vocabulario.cargar(CORPUS / "vocabulario.parquet", CORPUS / "ramas.parquet")
    ids_int, ids_sal = (vocab.id(t) for t in ("SINREG_INT", "SINREG_SAL"))
    val = CORPUS / "eventos_val"
    ds = {v: DatasetPretrain(val, vocab, _opciones(v == "v1"), augmentar=False) for v in ("v0", "v1")}
    alt = {(v, s): DatasetPretrain(val, vocab, _opciones(v == "v1", *((1.0, 0.0) if s == "inv" else (0.0, 1.0)), a.sop_v11),
                                   augmentar=False) for v in ("v0", "v1") for s in ("inv", "mez")}
    assert len(ds["v0"]) == len(ds["v1"])
    ks = np.arange(0, len(ds["v0"]), a.cada)
    docs = {v: [] for v in ("v0", "v1")}
    docs_alt = {v: [] for v in ("v0", "v1")}
    filas, trab, sop = [], [], []
    for w, k in enumerate(ks):
        k = int(k)
        i = ds["v0"].indices[k]
        c = ds["v0"].corpus
        ev0 = c.eventos(i, ds["v0"].fin[i])
        ev1 = con_grilla(ev0, fin_de_la_historia(FIN, None))
        d0, d1 = ds["v0"].documento(k), ds["v1"].documento(k)
        ini0, desc0 = _inicios(np.full(len(ev0), 6), len(d0))
        emp1 = np.flatnonzero(ev1.tipo == TIPO_EMPLEO)
        ini1_todos, desc1 = _inicios(np.where(ev1.tipo == TIPO_EMPLEO, 6, 2), len(d1))
        ini1 = ini1_todos[emp1]
        assert len(emp1) == len(ev0)

        # casos comunes: (evento de empleo j, campo f) presentes en los dos documentos y enmascarables; fondo
        cand = []  # (pos v0, pos v1, categoría, posición, evento j, campo f)
        for pos_fondo, cat in ((1, 5), (2, 6)):
            if d0.tokens[pos_fondo] >= N_ESPECIALES:
                cand.append((pos_fondo, pos_fondo, cat, POSICIONES.index("fondo"), -1, -1))
        for j in range(max(desc0, int(np.searchsorted(emp1, desc1))), len(ev0)):
            e = emp1[j]
            if e == 0:
                lugar = "primero"
            elif ev1.tipo[e - 1] == TIPO_INT:
                lugar = "tras_hueco"
            else:
                lugar = "nueva_sin_hueco" if ev0.nueva[j] else "continuidad"
            for f in range(5):
                p0, p1 = ini0[j] + f, ini1[j] + f
                assert d0.tokens[p0] == d1.tokens[p1]
                if d0.tokens[p0] >= N_ESPECIALES:
                    cand.append((p0, p1, f, POSICIONES.index(lugar), j, f))
        cand = np.array(cand, dtype=np.int64).reshape(-1, 6)
        rng = np.random.default_rng([77, k])
        n = min(int(np.floor(0.3 * len(d0))), len(cand))
        elegidos = cand[np.sort(rng.choice(len(cand), size=n, replace=False))]
        u, azar = rng.random(n), rng.integers(N_ESPECIALES, VOCAB_V0, size=n)

        # regla «copiar al vecino»: el mismo campo del evento visible más cercano (primero hacia atrás)
        oculto = np.zeros((len(ev0), 5), dtype=bool)
        emp_el = elegidos[elegidos[:, 4] >= 0]
        oculto[emp_el[:, 4], emp_el[:, 5]] = True
        primero_visible = max(desc0, int(np.searchsorted(emp1, desc1)))
        matriz = np.zeros((len(ev0), 5), dtype=np.int64)  # los eventos truncados quedan en cero (nunca se copian)
        matriz[desc0:] = d0.tokens[ini0[desc0:, None] + np.arange(5)]
        copia = np.full(n, -1, dtype=np.int64)
        for t, (p0, _, cat, _, j, f) in enumerate(elegidos):
            if j < 0:
                continue
            visibles = np.flatnonzero(~oculto[primero_visible:, f]) + primero_visible
            if len(visibles):
                atras = visibles[visibles < j]
                copia[t] = matriz[atras[-1] if len(atras) else visibles[visibles > j][0], f]

        for v, d, col in (("v0", d0, 0), ("v1", d1, 1)):
            pos = elegidos[:, col]
            orden = np.argsort(pos)
            assert (orden == np.arange(n)).all()  # el orden de los casos es el mismo en las dos versiones
            m = d.como_matriz()
            m[0] = _enmascarar_en(d.tokens, pos, u, azar)
            docs[v].append((m, pos, d.tokens[pos]))
        # tokens tapados del mismo evento: con más de uno, el modelo no sabe qué campo es cada uno (comparten fecha,
        # edad y segmento, y no hay posición dentro del evento), salvo por descarte de los visibles
        por_evento = np.bincount(emp_el[:, 4], minlength=len(ev0))
        for t in range(n):
            j = elegidos[t, 4]
            filas.append((w, elegidos[t, 2], elegidos[t, 3], copia[t], d0.tokens[elegidos[t, 0]],
                          por_evento[j] if j >= 0 else 0))
        n_hueco = int((ev1.tipo != TIPO_EMPLEO).sum())
        identico = len(d0) == len(d1) and (d0.como_matriz() == d1.como_matriz()).all()
        trab.append((k, n_hueco / max(len(ev1), 1), identico, len(ev0)))

        # SOP: un documento alterado por versión, enmascarado con la misma tasa sobre tokens de empleo
        estilo = "inv" if k % 2 == 0 else "mez"
        for v, normal in (("v0", d0), ("v1", d1)):
            da = alt[(v, estilo)].documento(k, np.random.default_rng([9, k]), con_sop=True)
            if da.sop == SOP_NORMAL:
                grupo = -1
                docs_alt[v].append(None)
            else:
                if np.array_equal(da.tokens, normal.tokens):
                    grupo = 0
                elif v == "v1" and _sinreg_fuera_de_lugar(da.tokens, ids_int, ids_sal):
                    grupo = 1
                else:
                    grupo = 2
                cand_a = np.flatnonzero((da.tokens >= N_ESPECIALES) & (da.tokens != ids_int) & (da.tokens != ids_sal))
                r = np.random.default_rng([78, k])
                na = min(n, len(cand_a))
                pos = np.sort(r.choice(cand_a, size=na, replace=False))
                m = da.como_matriz()
                m[0] = _enmascarar_en(da.tokens, pos, r.random(na), r.integers(N_ESPECIALES, VOCAB_V0, size=na))
                docs_alt[v].append((m, pos, da.tokens[pos]))
            sop.append((w, ("v0", "v1").index(v), grupo))
        if (w + 1) % 500 == 0:
            print(f"{w + 1}/{len(ks)} trabajadores", flush=True)

    a.salida.mkdir(parents=True, exist_ok=True)
    for v in ("v0", "v1"):
        _guardar_docs(a.salida / f"docs_{v}.npz", docs[v])
        _guardar_docs(a.salida / f"docs_{v}_sop.npz", docs_alt[v])
    f = np.array(filas, dtype=np.int64)
    t = np.array(trab, dtype=np.float64)
    np.savez(a.salida / "casos.npz", trabajador=f[:, 0], categoria=f[:, 1], posicion=f[:, 2], copia=f[:, 3],
             objetivo=f[:, 4], ocultos_evento=f[:, 5], k=t[:, 0].astype(np.int64), frac_hueco=t[:, 1], identico=t[:, 2].astype(bool),
             n_empleo=t[:, 3].astype(np.int64), sop=np.array(sop, dtype=np.int64))
    s = np.array(sop)
    print(f"listo: {len(ks)} trabajadores, {len(f)} casos ({t[:, 2].sum():.0f} documentos idénticos)")
    for v in (0, 1):
        g = s[s[:, 1] == v, 2]
        print(f"  SOP v{v}: alterables {np.mean(g >= 0):.3f}; " +
              ", ".join(f"{GRUPOS_SOP[x]} {np.mean(g[g >= 0] == x):.3f}" for x in range(3)))


def _guardar_docs(ruta: Path, docs: list) -> None:
    """Documentos de largo variable como arreglos planos con offsets; `None` = sin documento (largo 0)."""
    mats = [d[0] if d else np.zeros((4, 0), np.int64) for d in docs]
    pos = [d[1] if d else np.zeros(0, np.int64) for d in docs]
    obj = [d[2] if d else np.zeros(0, np.int64) for d in docs]
    np.savez(ruta, ids=np.concatenate(mats, axis=1).astype(np.int32),
             off=np.concatenate([[0], np.cumsum([m.shape[1] for m in mats])]),
             pos=np.concatenate(pos).astype(np.int32), off_pos=np.concatenate([[0], np.cumsum([len(p) for p in pos])]),
             obj=np.concatenate(obj).astype(np.int32))


# ---- inferir -------------------------------------------------------------------------------------------------------
def _modelo(corrida: Path) -> TransformerEncoder:
    ck = torch.load(corrida / "modelo_final.pt", map_location="cpu", weights_only=False)
    hp = hparams(**{**ck["hparams"], "sop_pesos": tuple(ck["hparams"]["sop_pesos"])})
    modelo = TransformerEncoder(hp)
    modelo.load_state_dict(ck["state_dict"])
    return modelo.eval()


def _pasar(modelo, ruta: Path, lote: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.load(ruta)
    off, off_pos = z["off"], z["off_pos"]
    n = len(off) - 1
    largos = np.diff(off)
    nll = np.full(len(z["obj"]), np.nan, dtype=np.float32)
    acierto = np.zeros(len(z["obj"]), dtype=bool)
    pred = np.zeros(len(z["obj"]), dtype=np.int32)
    p_normal = np.full(n, np.nan, dtype=np.float32)
    validos = np.flatnonzero(largos > 0)
    orden = validos[np.argsort(largos[validos], kind="stable")]
    with torch.no_grad():
        for b in range(0, len(orden), lote):
            grupo = orden[b:b + lote]
            muestras = [{"input_ids": z["ids"][:, off[i]:off[i + 1]].astype(np.int64),
                         "target_pos": z["pos"][off_pos[i]:off_pos[i + 1]].astype(np.int64),
                         "target_tokens": z["obj"][off_pos[i]:off_pos[i + 1]].astype(np.int64), "target_sop": 0}
                        for i in grupo]
            batch = collate(muestras, max_length=2560)
            mlm, sop = modelo(batch)
            p_normal[grupo] = sop.float().softmax(-1)[:, 0].numpy()
            t = batch["target_tokens"]
            perd = F.cross_entropy(mlm.float().permute(0, 2, 1), t, reduction="none", ignore_index=0).numpy()
            arg = mlm.argmax(-1)
            ok = (arg == t).numpy()
            for j, i in enumerate(grupo):
                m = off_pos[i + 1] - off_pos[i]
                nll[off_pos[i]:off_pos[i + 1]] = perd[j, :m]
                acierto[off_pos[i]:off_pos[i + 1]] = ok[j, :m]
                pred[off_pos[i]:off_pos[i + 1]] = arg[j, :m].numpy()
            if (b // lote) % 50 == 0:
                print(f"  {ruta.name}: {b + len(grupo)}/{len(orden)}", flush=True)
    return nll, acierto, pred, p_normal


def inferir(a) -> None:
    torch.set_num_threads(a.hilos)
    modelo = _modelo(a.corrida)
    nombre = a.nombre or a.corrida.name
    nll, acierto, pred, p_normal = _pasar(modelo, a.salida / f"docs_{a.version}.npz", a.lote)
    *_, p_alt = _pasar(modelo, a.salida / f"docs_{a.version}_sop.npz", a.lote)
    np.savez(a.salida / f"res_{nombre}.npz", version=a.version, nll=nll, acierto=acierto, pred=pred, p_normal=p_normal,
             p_normal_alt=p_alt)
    print(f"listo {nombre}: NLL media {np.nanmean(nll):.4f}")


# ---- resumir -------------------------------------------------------------------------------------------------------
def _sumas(casos, res, sel) -> tuple[np.ndarray, np.ndarray]:
    """Cantidad de casos y suma de NLL por trabajador, sobre los casos seleccionados."""
    n_trab = len(casos["k"])
    w = casos["trabajador"][sel]
    return np.bincount(w, minlength=n_trab).astype(float), np.bincount(w, res["nll"][sel], minlength=n_trab)


def _diferencia(casos, ra, rb, sel, trab_sel, rng, B) -> dict:
    """NLL de A y de B sobre los mismos casos (cociente de sumas), B − A con intervalo por remuestreo de trabajadores."""
    n, sa = _sumas(casos, ra, sel)
    _, sb = _sumas(casos, rb, sel)
    idx = np.flatnonzero(trab_sel & (n > 0))
    if len(idx) == 0:
        return {"trabajadores": 0}
    n, sa, sb = n[idx], sa[idx], sb[idx]
    r = rng.integers(0, len(idx), size=(B, len(idx)))
    boot = (sb[r].sum(1) - sa[r].sum(1)) / n[r].sum(1)
    return {"trabajadores": len(idx), "casos": int(n.sum()), "nll_a": float(sa.sum() / n.sum()),
            "nll_b": float(sb.sum() / n.sum()), "dif": float((sb.sum() - sa.sum()) / n.sum()),
            "ic": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
            "frac_peor": float(np.mean(sb / n > sa / n)), "dif_media_por_trabajador": float(np.mean((sb - sa) / n))}


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(puntaje de un alterado > puntaje de un normal), con empates a medias (Mann-Whitney)."""
    todos = np.concatenate([pos, neg])
    orden = todos.argsort(kind="stable")
    rangos = np.empty(len(todos))
    rangos[orden] = np.arange(1, len(todos) + 1)
    _, inv, cuenta = np.unique(todos, return_inverse=True, return_counts=True)
    suma = np.bincount(inv, rangos)
    rangos = (suma / cuenta)[inv]
    return float((rangos[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _auc_sop(casos, res, version: int, rng, B) -> dict:
    s = casos["sop"]
    s = s[s[:, 1] == version]
    trab, grupo = s[:, 0], s[:, 2]
    p_norm, p_alt = res["p_normal"], res["p_normal_alt"]
    neg = 1 - p_norm
    out = {}
    n_trab = len(casos["k"])
    for nombre, sel in (("todos", grupo >= 0), *((g, grupo == i) for i, g in enumerate(GRUPOS_SOP))):
        if sel.sum() == 0:
            continue
        w_alt = trab[sel]
        pos = 1 - p_alt[w_alt]
        boot = []
        for _ in range(B):
            r = rng.integers(0, n_trab, size=n_trab)
            peso = np.bincount(r, minlength=n_trab)
            pa = np.repeat(pos, peso[w_alt])
            boot.append(_auc(pa, np.repeat(neg, peso)) if len(pa) else np.nan)
        out[nombre] = {"alterados": int(sel.sum()), "auc": _auc(pos, neg),
                       "ic": [float(np.nanquantile(boot, 0.025)), float(np.nanquantile(boot, 0.975))]}
    return out


def _sop_pareado(casos, res, version: int, rng, B) -> dict:
    """AUC pareada: P(el alterado recibe menos P(normal) que el documento normal de la MISMA persona), empates a medias.
    Contra todos los normales (`_auc_sop`) se mezcla el tipo de persona: los alterados «invisibles», idénticos al normal,
    dan ~0,8 en lugar de 0,5."""
    s = casos["sop"]
    s = s[s[:, 1] == version]
    w, grupo = s[:, 0], s[:, 2]
    pn, pa = res["p_normal"][w], res["p_normal_alt"][w]
    gana = (pn > pa) + 0.5 * (pn == pa)
    out = {}
    for nombre, sel in (("todos", grupo >= 0), *((g, grupo == i) for i, g in enumerate(GRUPOS_SOP))):
        x = gana[sel]
        if len(x):
            boot = x[rng.integers(0, len(x), size=(B, len(x)))].mean(1)
            out[nombre] = {"alterados": len(x), "auc": float(x.mean()),
                           "ic": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]}
    return out


def resumir(a) -> None:
    casos = dict(np.load(a.salida / "casos.npz"))
    res = {m: dict(np.load(a.salida / f"res_{m}.npz")) for m in a.modelos}
    rng = np.random.default_rng(0)
    B = a.remuestreos
    base = a.base
    emp = casos["categoria"] < len(CAMPOS)
    todos_trab = np.ones(len(casos["k"]), dtype=bool)
    out = {"trabajadores": len(casos["k"]), "casos": len(casos["categoria"]), "base": base, "comparaciones": {}}

    for m in a.modelos:
        if m == base:
            continue
        c = {}
        # D2 (y D1 cuando las dos son de v0): por categoría y empleo en total
        c["por_categoria"] = {cat: _diferencia(casos, res[base], res[m], casos["categoria"] == i, todos_trab, rng, B)
                              for i, cat in enumerate(CATEGORIAS)}
        c["empleo"] = _diferencia(casos, res[base], res[m], emp, todos_trab, rng, B)
        # D3: documentos idénticos en las dos codificaciones
        c["empleo_identicos"] = _diferencia(casos, res[base], res[m], emp, casos["identico"], rng, B)
        # D4: estratos por peso de los huecos de la persona y por posición del caso en la trayectoria
        cortes = [0, 1e-9, 0.1, 0.3, 0.5, 0.7, 1.0001]
        c["por_frac_hueco"] = {f"{cortes[i]:.1f}-{cortes[i + 1]:.1f}" if i else "0":
                               _diferencia(casos, res[base], res[m], emp,
                                           (casos["frac_hueco"] >= cortes[i]) & (casos["frac_hueco"] < cortes[i + 1]),
                                           rng, B) for i in range(len(cortes) - 1)}
        c["por_posicion"] = {p: _diferencia(casos, res[base], res[m], emp & (casos["posicion"] == i), todos_trab, rng, B)
                             for i, p in enumerate(POSICIONES[:-1])}
        # D5: casos que la regla «copiar al vecino» acierta y los que no
        copia_ok = casos["copia"] == casos["objetivo"]
        c["por_ocultos_evento"] = {str(x): _diferencia(casos, res[base], res[m], emp & (np.minimum(casos["ocultos_evento"], 3) == x),
                                                    todos_trab, rng, B) for x in (1, 2, 3)}
        c["por_copia"] = {"copiable": _diferencia(casos, res[base], res[m], emp & copia_ok, todos_trab, rng, B),
                          "no_copiable": _diferencia(casos, res[base], res[m], emp & ~copia_ok, todos_trab, rng, B)}
        out["comparaciones"][f"{m} - {base}"] = c

    copia_ok = casos["copia"] == casos["objetivo"]
    out["copia"] = {cat: {"acierto_regla": float(copia_ok[casos["categoria"] == i].mean()),
                          **{f"acierto_{m}": float(res[m]["acierto"][casos["categoria"] == i].mean()) for m in a.modelos}}
                    for i, cat in enumerate(CAMPOS)}
    # campo de la predicción: ¿el error es de valor (campo correcto) o de campo (predice un valor de otra variable)?
    vocab = Vocabulario.cargar(CORPUS / "vocabulario.parquet", CORPUS / "ramas.parquet")
    cat_tok = np.array([CATEGORIAS.index("rama") if c == "rama2" else (CATEGORIAS.index(c) if c in CATEGORIAS else -1)
                        for c in vocab.categorias])
    ocultos = np.minimum(casos["ocultos_evento"], 3)
    out["campo"] = {m: {str(x): {"casos": int((emp & (ocultos == x)).sum()),
                                 "acierto": float(res[m]["acierto"][emp & (ocultos == x)].mean()),
                                 "campo_equivocado": float((cat_tok[res[m]["pred"]] != casos["categoria"])[emp & (ocultos == x)].mean())}
                        for x in (1, 2, 3)} for m in a.modelos}
    out["sop"] = {m: _auc_sop(casos, res[m], 0 if str(res[m]["version"]) == "v0" else 1, rng, a.remuestreos_auc)
                  for m in a.modelos}
    out["sop_pareado"] = {m: _sop_pareado(casos, res[m], 0 if str(res[m]["version"]) == "v0" else 1, rng, B)
                          for m in a.modelos}
    (a.salida / "resumen.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    _imprimir(out)


def _fmt(d: dict) -> str:
    if not d.get("trabajadores"):
        return "—"
    return f"{d['nll_a']:.3f} → {d['nll_b']:.3f}  {d['dif']:+.3f} [{d['ic'][0]:+.3f}, {d['ic'][1]:+.3f}]  peor {100 * d['frac_peor']:.0f} %"


def _imprimir(out: dict) -> None:
    print(f"{out['trabajadores']} trabajadores, {out['casos']} casos; base {out['base']}")
    for nombre, c in out["comparaciones"].items():
        print(f"\n== {nombre} ==")
        print(f"  empleo:            {_fmt(c['empleo'])}  ({c['empleo']['casos']} casos)")
        print(f"  empleo idénticos:  {_fmt(c['empleo_identicos'])}  ({c['empleo_identicos'].get('trabajadores', 0)} trab.)")
        for k, v in c["por_categoria"].items():
            print(f"  {k:17s}  {_fmt(v)}")
        for k, v in c["por_frac_hueco"].items():
            print(f"  huecos {k:10s}  {_fmt(v)}  ({v.get('trabajadores', 0)} trab.)")
        for k, v in {**c["por_posicion"], **{f"ocultos_ev {x}": v for x, v in c["por_ocultos_evento"].items()}, **c["por_copia"]}.items():
            print(f"  {k:17s}  {_fmt(v)}  ({v.get('casos', 0)} casos)")
    print("\nAcierto de la regla «copiar al vecino» y de los modelos")
    for k, v in out["copia"].items():
        print(f"  {k:15s} " + "  ".join(f"{kk[8:] or kk}: {vv:.3f}" for kk, vv in v.items()))
    print("\nAcierto y predicciones de otro campo, según los tokens tapados del mismo evento (1, 2, 3 o más)")
    for m, g in out["campo"].items():
        print(f"  {m}: " + "; ".join(f"{x}: acierto {v['acierto']:.3f}, otro campo {v['campo_equivocado']:.3f} (n={v['casos']})"
                                     for x, v in g.items()))
    for clave, titulo in (("sop_pareado", "pareada: contra el normal de la misma persona"),
                          ("sop", "contra todos los normales; mezcla el tipo de persona")):
        print(f"\nAUC del SOP por grupo de alterados ({titulo})")
        for m, g in out[clave].items():
            print(f"  {m}: " + "; ".join(f"{k} {v['auc']:.3f} [{v['ic'][0]:.3f}, {v['ic'][1]:.3f}] (n={v['alterados']})"
                                         for k, v in g.items()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--salida", type=Path, default=SALIDA)
    sub = ap.add_subparsers(dest="paso", required=True)
    p = sub.add_parser("preparar")
    p.add_argument("--cada", type=int, default=20)
    p.add_argument("--sop-v11", action="store_true",
                   help="alteraciones del SOP de v1.1 para la codificación con grilla (los SINREG no se mueven)")
    p = sub.add_parser("inferir")
    p.add_argument("corrida", type=Path)
    p.add_argument("version", choices=("v0", "v1"))
    p.add_argument("--nombre")
    p.add_argument("--lote", type=int, default=16)
    p.add_argument("--hilos", type=int, default=6)
    p = sub.add_parser("resumir")
    p.add_argument("modelos", nargs="+", help="nombres de `res_<nombre>.npz`; el primero de v0 hace de base")
    p.add_argument("--base", default="v0_s0")
    p.add_argument("--remuestreos", type=int, default=2000)
    p.add_argument("--remuestreos-auc", type=int, default=300)
    a = ap.parse_args()
    {"preparar": preparar, "inferir": inferir, "resumir": resumir}[a.paso](a)


if __name__ == "__main__":
    main()
