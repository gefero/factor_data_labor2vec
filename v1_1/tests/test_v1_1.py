"""Pruebas de v1.1: el pre-entrenamiento corregido sobre la codificación de v1 (plans/PLAN_v1_1.md)."""

import numpy as np
import pytest

from labor2vec.tokens.datos import CorpusEnMemoria, DatasetPretrain
from labor2vec.tokens.documento import (
    N_FONDO,
    SOP_INVERTIDO,
    SOP_MEZCLADO,
    SOP_NORMAL,
    Eventos,
    Opciones,
    con_grilla,
    construir_documento,
    enmascarar,
    enmascarar_con,
    opciones_de,
)
from labor2vec.tokens.linea_base import logp_unigrama
from labor2vec.tokens.vocabulario import SEP

V11 = dict(sop_sinreg_fijo=True, mlm_sinreg="entrada", mask_base="sin_sinreg")


def _trabajador_1(mundo):
    c = CorpusEnMemoria(mundo["salida"] / "eventos_train")
    i = int(np.flatnonzero(c.id_trabajador == 1)[0])
    return c, i  # meses 0, 1, 1, 2, 2, 3, 6: con fin_ventana=9, intermitencia en 4 y 5 y salida en 7 a 9


def _doc(mundo, rng=None, con_sop=False, **kw):
    v, (c, i) = mundo["vocab"], _trabajador_1(mundo)
    return construir_documento(c.eventos(i), c.sexo[i], c.fnacim[i], v, v.tablas(), Opciones(fin_ventana=9, **kw),
                               rng, con_sop=con_sop)


def _eventos(meses, rem):
    n = len(meses)
    return Eventos(mes_idx=np.array(meses, dtype=np.int16), pct_rem=np.array(rem, dtype=np.int8),
                   tramo=np.full(n, 2, dtype=np.int8), r34=np.full(n, 111, dtype=np.int16),
                   provi=np.full(n, 1, dtype=np.int8), empini=np.full(n, 3, dtype=np.int8),
                   nueva=np.zeros(n, dtype=bool))


def _sop(ev, mundo, clase, semilla, **kw):
    v = mundo["vocab"]
    p = dict(p_sop_invertido=1.0, p_sop_mezclado=0.0) if clase == SOP_INVERTIDO else dict(p_sop_invertido=0.0,
                                                                                       p_sop_mezclado=1.0)
    return construir_documento(ev, 1, 1980, v, v.tablas(), Opciones(**p, **kw), np.random.default_rng(semilla),
                               con_sop=True)


# ---- opciones ---------------------------------------------------------------------------------------------
def test_opciones_por_defecto_son_las_de_v1_y_se_leen_del_yaml():
    op = opciones_de({"max_length": 128, "fin_ventana": 10})
    assert (op.sop_sinreg_fijo, op.mlm_sinreg, op.mask_base) == (False, "objetivo", "largo")
    op = opciones_de({"max_length": 128, "fin_ventana": 10, **V11})
    assert (op.sop_sinreg_fijo, op.mlm_sinreg, op.mask_base) == (True, "entrada", "sin_sinreg")
    for malo in ({"mlm_sinreg": "nada"}, {"mask_base": "otro"}):
        with pytest.raises(ValueError):
            opciones_de({"max_length": 128, "fin_ventana": 10, **malo})


# ---- SOP --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("clase", [SOP_INVERTIDO, SOP_MEZCLADO])
def test_sop_v1_1_no_mueve_los_sinreg_y_siempre_cambia_el_documento(mundo, clase):
    v = mundo["vocab"]
    sinreg = v.tablas()["sinreg"]
    base = _doc(mundo, grilla=True)
    es_sinreg = np.isin(base.tokens, sinreg)
    assert np.isin(base.tokens, sinreg[0]).any() and np.isin(base.tokens, sinreg[1]).any()  # tiene los dos tipos
    c, i = _trabajador_1(mundo)
    alterados = 0
    for semilla in range(20):
        p = dict(p_sop_invertido=1.0, p_sop_mezclado=0.0) if clase == SOP_INVERTIDO else dict(p_sop_invertido=0.0,
                                                                                           p_sop_mezclado=1.0)
        doc = construir_documento(c.eventos(i), c.sexo[i], c.fnacim[i], v, v.tablas(),
                                  Opciones(fin_ventana=9, grilla=True, **V11, **p), np.random.default_rng(semilla),
                                  con_sop=True)
        assert (doc.tokens[es_sinreg] == base.tokens[es_sinreg]).all()  # cada SINREG, de su tipo, en su mes
        assert np.array_equal(doc.tokens == SEP, base.tokens == SEP) and (doc.abspos == base.abspos).all()
        if doc.sop != SOP_NORMAL:
            alterados += 1
            assert doc.sop == clase and (doc.tokens != base.tokens).any()
        else:
            assert np.array_equal(doc.tokens, base.tokens)
    assert alterados == 20 if clase == SOP_MEZCLADO else alterados > 0


def test_sop_v1_1_no_altera_si_solo_se_podrian_mover_meses_sin_registro(mundo):
    """Un mes de empleo y varios de salida: en v1 se intercambiaban los SINREG_SAL (alteración invisible); en v1.1 no."""
    ev = con_grilla(_eventos([0], [50]), fin=5)
    v1 = _sop(ev, mundo, SOP_MEZCLADO, 0)
    assert v1.sop == SOP_MEZCLADO and np.array_equal(v1.tokens, _sop(ev, mundo, SOP_NORMAL, 0).tokens)  # v1: invisible
    for clase in (SOP_INVERTIDO, SOP_MEZCLADO):
        assert _sop(ev, mundo, clase, 0, **V11).sop == SOP_NORMAL


def test_sop_v1_1_no_altera_si_los_meses_de_empleo_son_iguales(mundo):
    ev = con_grilla(_eventos([0, 1, 3], [50, 50, 50]), fin=5)
    for clase in (SOP_INVERTIDO, SOP_MEZCLADO):
        assert _sop(ev, mundo, clase, 0, **V11).sop == SOP_NORMAL
    ev = con_grilla(_eventos([0, 1, 3], [50, 50, 70]), fin=5)
    for clase in (SOP_INVERTIDO, SOP_MEZCLADO):
        assert _sop(ev, mundo, clase, 0, **V11).sop == clase


def test_sop_v1_1_invertido_simetrico_queda_normal(mundo):
    """Invertir A B A no cambia nada: el documento no cuenta como alterado."""
    ev = _eventos([0, 1, 2], [50, 70, 50])
    assert _sop(ev, mundo, SOP_INVERTIDO, 0, **V11).sop == SOP_NORMAL
    assert _sop(ev, mundo, SOP_MEZCLADO, 0, **V11).sop == SOP_MEZCLADO


# ---- MLM --------------------------------------------------------------------------------------------------
def test_enmascarado_por_defecto_igual_al_de_v1(mundo):
    v = mundo["vocab"]
    doc = _doc(mundo, grilla=True)
    for s in range(10):
        a = enmascarar(doc.tokens, len(v), 0.3, np.random.default_rng(s))
        b = enmascarar_con(doc.tokens, len(v), Opciones(grilla=True), v.tablas()["sinreg"], np.random.default_rng(s))
        assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_mlm_v1_1_no_tapa_ni_inventa_sinreg(mundo):
    v = mundo["vocab"]
    sinreg = v.tablas()["sinreg"]
    doc = _doc(mundo, grilla=True)
    es_sinreg = np.isin(doc.tokens, sinreg)
    op = Opciones(grilla=True, mask_ratio=0.9, **V11)  # tasa alta: casi todo se tapa y hay muchos reemplazos al azar
    for s in range(50):
        tok, _, obj = enmascarar_con(doc.tokens, len(v), op, sinreg, np.random.default_rng(s))
        assert not np.isin(obj, sinreg).any()
        assert np.array_equal(np.isin(tok, sinreg), es_sinreg) and (tok[es_sinreg] == doc.tokens[es_sinreg]).all()


def test_mlm_v1_1_tapa_los_mismos_casos_que_v0(mundo):
    """Mismos candidatos, en el mismo orden, misma cantidad y mismo azar: los objetivos y la entrada visible de empleo
    coinciden con los del documento de v0 de la misma persona."""
    v = mundo["vocab"]
    sinreg = v.tablas()["sinreg"]
    d0, d11 = _doc(mundo), _doc(mundo, grilla=True)
    assert len(d11) > len(d0)
    no_sinreg = ~np.isin(d11.tokens, sinreg) & ~((d11.tokens == SEP) & np.roll(np.isin(d11.tokens, sinreg), 1))
    assert np.array_equal(d11.tokens[no_sinreg], d0.tokens)  # sacando los bloques SINREG queda el documento de v0
    for s in range(20):
        t0, p0, o0 = enmascarar(d0.tokens, len(v) - len(sinreg), 0.3, np.random.default_rng(s))  # vocabulario de v0
        t11, p11, o11 = enmascarar_con(d11.tokens, len(v), Opciones(grilla=True, **V11), sinreg, np.random.default_rng(s))
        assert np.array_equal(o0, o11)
        assert np.array_equal(t11[no_sinreg], t0)
        assert np.array_equal(np.flatnonzero(no_sinreg)[p0], p11)


def test_mlm_v1_1_sin_huecos_tapa_lo_mismo_que_v1(mundo):
    """Sin huecos, mismas posiciones y objetivos que v1; sólo cambian los reemplazos al azar, que en v1.1 no son SINREG."""
    v = mundo["vocab"]
    ev = _eventos([0, 1, 2, 3], [10, 20, 30, 40])
    doc = construir_documento(ev, 1, 1980, v, v.tablas(), Opciones(fin_ventana=3, grilla=True))
    for s in range(5):
        _, p1, o1 = enmascarar(doc.tokens, len(v), 0.3, np.random.default_rng(s))
        _, p11, o11 = enmascarar_con(doc.tokens, len(v), Opciones(grilla=True, **V11), v.tablas()["sinreg"],
                                     np.random.default_rng(s))
        assert np.array_equal(p1, p11) and np.array_equal(o1, o11)


def test_unigrama_sin_masa_en_sinreg(mundo):
    v = mundo["vocab"]
    sinreg = v.tablas()["sinreg"]
    ds = DatasetPretrain(mundo["salida"] / "eventos_train", v, Opciones(max_length=2560, fin_ventana=20, grilla=True),
                         augmentar=False)
    con = logp_unigrama(ds, len(v), 50)
    sin = logp_unigrama(ds, len(v), 50, excluir=sinreg)
    assert (con[sinreg] > -30).all() and (sin[sinreg] == -30).all()
    assert abs(float(sin.exp().sum()) - 1) < 1e-5


def test_dataset_v1_1_muestras_validas(mundo):
    v = mundo["vocab"]
    op = Opciones(max_length=2560, fin_ventana=20, grilla=True, **V11)
    ds = DatasetPretrain(mundo["salida"] / "eventos_train", v, op, augmentar=True)
    for k in range(len(ds)):
        m = ds[k]
        assert not np.isin(m["target_tokens"], v.tablas()["sinreg"]).any()
        assert m["input_ids"][0][:N_FONDO][0] != SEP


# ---- de punta a punta -------------------------------------------------------------------------------------
def test_pretrain_v1_1_registra_la_validacion_por_grupo(mundo, tmp_path):
    import csv

    import yaml

    from labor2vec import entrenar
    from labor2vec.entrenar import cargar_config
    from labor2vec.modelo.pretrain import GRUPOS, grupos_de_tokens

    g = grupos_de_tokens(mundo["vocab"])
    cats = mundo["vocab"].categorias
    assert {cats[i] for i in np.flatnonzero(g.numpy() == GRUPOS.index("sinreg"))} == {"sinreg"}
    assert {cats[i] for i in np.flatnonzero(g.numpy() == GRUPOS.index("fondo"))} == {"sexo", "nacimiento"}

    cfg = cargar_config("configs/humo_cpu.yaml", [
        f"datos.corpus={mundo['salida']}", "datos.tamano_lote=8", "datos.max_length=256", "datos.fin_ventana=30",
        "entrenamiento.epocas=1", "entrenamiento.pasos_por_epoca=2", "entrenamiento.val_lotes=2",
        "entrenamiento.guardar_cada_pasos=2", "entrenamiento.muestra_unigrama=20",
        f"entrenamiento.salida={tmp_path / 'pre'}"])
    assert all(cfg["datos"][k] == x for k, x in V11.items())
    ruta = tmp_path / "pre.yaml"
    ruta.write_text(yaml.safe_dump(cfg))
    entrenar.main(["--config", str(ruta)])
    columnas = set()
    for f in (tmp_path / "pre").rglob("metrics.csv"):
        with open(f) as fh:
            columnas |= set(next(csv.reader(fh)))
    assert {"val/mlm_loss_empleo", "val/mlm_acc_empleo", "val/mlm_loss_fondo"} <= columnas
    assert "val/mlm_loss_sinreg" not in columnas
