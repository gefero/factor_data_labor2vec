"""Pruebas de v1: grilla mensual con huecos explícitos (SINREG_INT y SINREG_SAL), plans/PLAN_v1.md."""

import numpy as np
import pyarrow.parquet as pq
import pytest

from labor2vec.tokens.datos import CorpusEnMemoria, DatasetPretrain
from labor2vec.tokens.documento import (FIN_PANEL, N_FONDO, SOP_INVERTIDO, SOP_MEZCLADO, SOP_NORMAL, TIPO_EMPLEO,
                                        TIPO_INT, TIPO_SAL, Eventos, Opciones, _permutacion, con_grilla,
                                        construir_documento, dias_de_mes, enmascarar, fin_de_la_historia, opciones_de)
from labor2vec.tokens.vocabulario import CATEGORIA_SINREG, N_ESPECIALES, SEP, TOKENS_SINREG, Vocabulario


def _eventos(meses):
    n = len(meses)
    return Eventos(mes_idx=np.array(meses, dtype=np.int16), pct_rem=np.arange(n, dtype=np.int8),
                   tramo=np.full(n, 2, dtype=np.int8), r34=np.full(n, 111, dtype=np.int16),
                   provi=np.full(n, 1, dtype=np.int8), empini=np.full(n, 3, dtype=np.int8),
                   nueva=np.zeros(n, dtype=bool))


def _trabajador_1(mundo):
    c = CorpusEnMemoria(mundo["salida"] / "eventos_train")
    i = int(np.flatnonzero(c.id_trabajador == 1)[0])
    return c, i  # meses 0, 1, 1, 2, 2, 3, 6: hueco en los meses 4 y 5


def _doc(mundo, **kw):
    v, (c, i) = mundo["vocab"], _trabajador_1(mundo)
    return construir_documento(c.eventos(i), c.sexo[i], c.fnacim[i], v, v.tablas(), Opciones(grilla=True, **kw))


def _bloques(doc, mundo):
    """Lista de (nombre del primer token, fecha) de cada bloque después del background, partiendo por [SEP]."""
    t, d, nombres = doc.tokens[N_FONDO:], doc.abspos[N_FONDO:], mundo["vocab"].tokens
    fin = np.flatnonzero(t == SEP)
    ini = np.concatenate([[0], fin[:-1] + 1])
    return [(nombres[t[a]], int(d[a]), int(b - a + 1)) for a, b in zip(ini, fin)]


# ---- vocabulario ------------------------------------------------------------------------------------------
def test_tokens_sinreg_van_al_final_y_no_se_guardan(mundo):
    v = mundo["vocab"]
    assert v.tokens[-2:] == list(TOKENS_SINREG) and v.categorias[-2:] == [CATEGORIA_SINREG] * 2
    assert list(v.tablas()["sinreg"]) == [len(v) - 2, len(v) - 1]
    guardado = pq.read_table(mundo["salida"] / "vocabulario.parquet").to_pydict()
    assert not set(TOKENS_SINREG) & set(guardado["token"]) and len(guardado["token"]) == len(v) - 2
    c = mundo["salida"]
    assert Vocabulario.cargar(c / "vocabulario.parquet", c / "ramas.parquet").tokens == v.tokens


def test_ids_de_v0_no_se_mueven(mundo):
    """Los 2 tokens nuevos van detrás de todo: un vocabulario de v0 y el de v1 coinciden en los ids de los demás."""
    v = mundo["vocab"]
    guardado = pq.read_table(mundo["salida"] / "vocabulario.parquet").sort_by("id").to_pydict()
    # el archivo conserva los nombres de v0 (ANT_k): mismos ids, sólo cambia el rótulo de las 4 empresas
    assert guardado["token"] == [t.replace("EMPINI_", "ANT_") for t in v.tokens[:-2]]


def test_inicio_de_la_empresa_se_llama_empini_pero_el_archivo_conserva_los_nombres_de_v0(mundo):
    """`tr_anio_ini` es la edad de la empresa, no la antigüedad del trabajador: en v1 son EMPINI_k; el archivo (el de v0,
    compartido) sigue con ANT_k / «antigüedad», que `cargar` traduce."""
    v = mundo["vocab"]
    assert [t for t in v.tokens if t.startswith("EMPINI_")] == [f"EMPINI_{k}" for k in range(1, 5)]
    assert not any(t.startswith("ANT_") for t in v.tokens) and "antigüedad" not in v.categorias
    assert v.categorias.count("inicio_empresa") == 4
    guardado = pq.read_table(mundo["salida"] / "vocabulario.parquet").to_pydict()
    assert [t for t in guardado["token"] if t.startswith("ANT_")] == [f"ANT_{k}" for k in range(1, 5)]
    assert "antigüedad" in guardado["categoria"] and "inicio_empresa" not in guardado["categoria"]
    assert not any(t.startswith("EMPINI_") for t in guardado["token"])


# ---- grilla -----------------------------------------------------------------------------------------------
def test_con_grilla_tipos_orden_y_sin_relleno_inicial():
    ev = con_grilla(_eventos([3, 4, 4, 7, 8]), fin=11)
    assert list(ev.mes_idx) == [3, 4, 4, 5, 6, 7, 8, 9, 10, 11]
    assert list(ev.tipo) == [0, 0, 0, TIPO_INT, TIPO_INT, 0, 0, TIPO_SAL, TIPO_SAL, TIPO_SAL]
    # los eventos de empleo conservan su orden y sus datos; los huecos no tienen atributos
    assert list(ev.pct_rem[ev.tipo == TIPO_EMPLEO]) == [0, 1, 2, 3, 4]
    assert (ev.pct_rem[ev.tipo != TIPO_EMPLEO] == -1).all() and (ev.r34[ev.tipo != TIPO_EMPLEO] == -1).all()


def test_con_grilla_sin_huecos_o_vacio_no_cambia():
    ev = _eventos([5, 6, 6, 7])
    assert con_grilla(ev, fin=7) is ev
    assert len(con_grilla(_eventos([]), fin=7)) == 0


def test_el_tipo_depende_de_la_ventana():
    ev = _eventos([0, 1, 5])
    ancha = con_grilla(ev, fin=8)  # el hueco de los meses 2-4 se cierra con el registro del mes 5
    assert list(ancha.tipo[(ancha.mes_idx >= 2) & (ancha.mes_idx <= 4)]) == [TIPO_INT] * 3
    corta = con_grilla(ev.tomar(ev.mes_idx <= 3), fin=4)  # ventana que termina antes de la vuelta
    assert list(corta.tipo[corta.mes_idx >= 2]) == [TIPO_SAL] * 3


def test_fin_de_la_historia():
    assert fin_de_la_historia(239, None) == 239 and fin_de_la_historia(None, None) == FIN_PANEL
    assert fin_de_la_historia(239, "sin_ultimo_anio") == 227 and fin_de_la_historia(239, "atrasar_2_anios") == 215
    assert fin_de_la_historia(239, "salario_arriba") == 239


# ---- documento --------------------------------------------------------------------------------------------
def test_documento_con_huecos_e_intermitencia_y_salida(mundo):
    doc = _doc(mundo, fin_ventana=9)
    # eventos de empleo en los meses 0, 1, 1, 2, 2, 3 y 6; huecos 4 y 5 (INT) y 7, 8 y 9 (SAL)
    nombres = [(n[:6] == "SINREG", n if n[:6] == "SINREG" else "", f, k) for n, f, k in _bloques(doc, mundo)]
    esperado = ([False] * 6 + [True, True, False, True, True, True])
    assert [b[0] for b in nombres] == esperado
    assert [b[1] for b in nombres if b[0]] == ["SINREG_INT"] * 2 + ["SINREG_SAL"] * 3
    assert [b[3] for b in nombres] == [6] * 6 + [2, 2, 6, 2, 2, 2]
    assert len(doc) == N_FONDO + 6 * 7 + 2 * 5
    # cada hueco lleva la fecha del mes que representa, y las fechas no retroceden
    fechas = [b[2] for b in nombres]
    assert fechas[6:8] == list(dias_de_mes(np.array([4, 5]))) and fechas[9:] == list(dias_de_mes(np.array([7, 8, 9])))
    assert np.all(np.diff(doc.abspos[N_FONDO:]) >= 0) and np.all(np.diff(doc.edad[N_FONDO:]) >= 0)
    # el ciclo de segmentos sigue por evento, también en los huecos
    inicios = np.flatnonzero(np.concatenate([[True], doc.tokens[N_FONDO:-1] == SEP])) + N_FONDO
    assert list(doc.segmento[inicios]) == [2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1]


def test_el_hueco_cambia_de_tipo_con_la_ventana(mundo):
    larga = [n for n, _, _ in _bloques(_doc(mundo, fin_ventana=9), mundo) if n.startswith("SINREG")]
    corta = [n for n, _, _ in _bloques(_doc(mundo, fin_ventana=5), mundo) if n.startswith("SINREG")]
    assert larga[:2] == ["SINREG_INT"] * 2 and corta == ["SINREG_SAL"] * 2  # con la ventana a 5 el mes 6 no existe


def test_sin_ventana_la_grilla_llega_al_fin_del_panel(mundo):
    doc = _doc(mundo)
    assert doc.abspos[-1] == dias_de_mes(np.array([FIN_PANEL]))[0]
    assert len(doc) == N_FONDO + 6 * 7 + 2 * (2 + (FIN_PANEL - 6))


def test_trabajador_sin_huecos_queda_igual_que_en_v0(mundo):
    v, c = mundo["vocab"], CorpusEnMemoria(mundo["salida"] / "eventos_train")
    for i in range(len(c)):
        ev = c.eventos(i)
        meses = np.unique(ev.mes_idx)
        if len(meses) and meses[-1] - meses[0] + 1 == len(meses):
            ini = Opciones(fin_ventana=int(meses[-1]))
            a = construir_documento(ev, c.sexo[i], c.fnacim[i], v, v.tablas(), ini)
            b = construir_documento(ev, c.sexo[i], c.fnacim[i], v, v.tablas(), Opciones(fin_ventana=int(meses[-1]),
                                                                                     grilla=True))
            assert np.array_equal(a.tokens, b.tokens) and np.array_equal(a.abspos, b.abspos)
            return
    pytest.fail("el panel sintético debería tener algún trabajador sin huecos")


def test_sinreg_uno_usa_un_solo_token(mundo):
    doc = _doc(mundo, fin_ventana=9, sinreg="uno")
    assert {n for n, _, _ in _bloques(doc, mundo) if n.startswith("SINREG")} == {"SINREG_INT"}
    with pytest.raises(ValueError):
        _doc(mundo, fin_ventana=9, sinreg="tres")


def test_intervencion_sin_ultimo_anio_no_inventa_salidas(mundo):
    """Al borrar el último año la historia termina 12 meses antes: no hay meses SAL hasta el fin de la ventana."""
    doc = _doc(mundo, fin_ventana=17, intervencion="sin_ultimo_anio")
    b = _bloques(doc, mundo)
    assert [n for n, _, _ in b if n.startswith("SINREG")] == ["SINREG_SAL"] * 2  # meses 4 y 5 (después del 3, sin vuelta)
    assert b[-1][1] == dias_de_mes(np.array([5]))[0]


def test_sop_con_grilla_no_mueve_los_sep_ni_cambia_de_largo_los_bloques(mundo):
    """La perturbación intercambia sólo bloques del mismo largo: los [SEP], fechas y segmentos quedan donde estaban."""
    v, (c, i) = mundo["vocab"], _trabajador_1(mundo)
    sinreg = v.tablas()["sinreg"]
    base = _doc(mundo, fin_ventana=9)
    for p, clase in ((dict(p_sop_invertido=1.0, p_sop_mezclado=0.0), SOP_INVERTIDO),
                     (dict(p_sop_invertido=0.0, p_sop_mezclado=1.0), SOP_MEZCLADO)):
        for semilla in range(10):
            doc = construir_documento(c.eventos(i), c.sexo[i], c.fnacim[i], v, v.tablas(),
                                      Opciones(grilla=True, fin_ventana=9, **p), np.random.default_rng(semilla),
                                      con_sop=True)
            assert doc.sop == clase
            assert np.array_equal(doc.tokens == SEP, base.tokens == SEP)
            assert np.array_equal(np.isin(doc.tokens, sinreg), np.isin(base.tokens, sinreg))  # los huecos siguen en su lugar
            assert (doc.abspos == base.abspos).all() and (doc.segmento == base.segmento).all()
            assert sorted(doc.tokens) == sorted(base.tokens) and (doc.tokens != base.tokens).any()


def test_sop_sin_dos_bloques_del_mismo_largo_no_perturba(mundo):
    v = mundo["vocab"]
    ev = con_grilla(_eventos([0]), fin=1)  # un mes con empleo y uno de salida: bloques de 6 y de 2 tokens
    doc = construir_documento(ev, 1, 1980, v, v.tablas(), Opciones(p_sop_invertido=1.0, p_sop_mezclado=0.0),
                              np.random.default_rng(0), con_sop=True)
    assert doc.sop == SOP_NORMAL


def test_sop_sin_grilla_es_identico_al_algoritmo_de_v0(mundo):
    """Con un solo largo de bloque se consume el mismo azar y se obtiene el mismo orden que antes."""
    v, (c, i) = mundo["vocab"], _trabajador_1(mundo)
    base = construir_documento(c.eventos(i), c.sexo[i], c.fnacim[i], v, v.tablas(), Opciones())
    n = (len(base) - N_FONDO) // 6
    for semilla in range(5):
        rng = np.random.default_rng(semilla)
        u = rng.random()
        orden = np.arange(n)[::-1] if u < 0.5 else _permutacion(rng, n)
        esperado = np.concatenate([base.tokens[:N_FONDO]] + [base.tokens[N_FONDO + 6 * j: N_FONDO + 6 * j + 6] for j in orden])
        doc = construir_documento(c.eventos(i), c.sexo[i], c.fnacim[i], v, v.tablas(),
                                  Opciones(p_sop_invertido=0.5, p_sop_mezclado=0.5), np.random.default_rng(semilla),
                                  con_sop=True)
        assert np.array_equal(doc.tokens, esperado)


def test_mlm_puede_enmascarar_sinreg(mundo):
    doc = _doc(mundo, fin_ventana=9)
    es_sinreg = np.isin(doc.tokens, mundo["vocab"].tablas()["sinreg"])
    assert (doc.tokens[es_sinreg] >= N_ESPECIALES).all()
    visto = False
    for s in range(30):
        _, _, obj = enmascarar(doc.tokens, len(mundo["vocab"]), 0.3, np.random.default_rng(s))
        visto |= bool(np.isin(obj, mundo["vocab"].tablas()["sinreg"]).any())
    assert visto


def test_opciones_de_lee_grilla_y_sinreg():
    op = opciones_de({"max_length": 128, "fin_ventana": 10})
    assert (op.grilla, op.sinreg) == (False, "dos")
    op = opciones_de({"max_length": 128, "fin_ventana": 10, "grilla": True, "sinreg": "uno"})
    assert (op.grilla, op.sinreg) == (True, "uno")


# ---- dataset ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("kw", [dict(fin_ventana=10), dict(fin_ventana=None), dict(fin_ventana=10, atributos="sin_ultimo_valor"),
                                dict(fin_ventana=10, eventos_compactos=True), dict(fin_ventana=40)])
def test_largos_coincide_con_el_documento_con_grilla(mundo, kw):
    op = Opciones(max_length=2560, grilla=True, **kw)
    ds = DatasetPretrain(mundo["salida"] / "eventos_train", mundo["vocab"], op, augmentar=False)
    largos = [len(ds.documento(k, ds._generador(k), False)) for k in range(len(ds))]
    assert largos == ds.largos().tolist()
    assert max(largos) > N_FONDO + 6  # alguien tiene huecos y largos distintos
    sin = DatasetPretrain(mundo["salida"] / "eventos_train", mundo["vocab"], Opciones(max_length=2560, **kw),
                          augmentar=False)
    assert (ds.largos() >= sin.largos()).all() and (ds.largos() > sin.largos()).any()


def test_dataset_con_grilla_devuelve_muestras_validas(mundo):
    op = Opciones(max_length=2560, fin_ventana=20, grilla=True)
    ds = DatasetPretrain(mundo["salida"] / "eventos_train", mundo["vocab"], op, augmentar=True)
    for k in range(min(10, len(ds))):
        m = ds[k]
        assert m["input_ids"].shape[0] == 4 and m["input_ids"][0].max() < len(mundo["vocab"])
        assert len(m["target_pos"]) == len(m["target_tokens"])


# ---- de punta a punta -------------------------------------------------------------------------------------
def test_pretrain_con_la_configuracion_de_v1_usa_la_grilla(mundo, tmp_path):
    """`humo_cpu.yaml` trae `grilla: true`: el pre-entrenamiento corre y los documentos tienen tokens SINREG."""
    from labor2vec import entrenar
    from labor2vec.entrenar import cargar_config

    cfg = cargar_config("configs/humo_cpu.yaml", [
        f"datos.corpus={mundo['salida']}", "datos.tamano_lote=8", "datos.max_length=256", "datos.fin_ventana=30",
        "entrenamiento.epocas=1", "entrenamiento.pasos_por_epoca=2", "entrenamiento.val_lotes=2",
        "entrenamiento.guardar_cada_pasos=2", "entrenamiento.muestra_unigrama=20",
        f"entrenamiento.salida={tmp_path / 'pre'}"])
    assert cfg["datos"]["grilla"] is True
    import yaml
    ruta = tmp_path / "pre.yaml"
    ruta.write_text(yaml.safe_dump(cfg))
    entrenar.main(["--config", str(ruta)])
    assert (tmp_path / "pre" / "modelo_final.pt").exists()
    dm, _ = entrenar.preparar(cfg)
    ds = dm.ds["train"]
    sinreg = mundo["vocab"].tablas()["sinreg"]
    assert any(np.isin(ds.documento(k, ds._generador(k), False).tokens, sinreg).any() for k in range(len(ds)))
