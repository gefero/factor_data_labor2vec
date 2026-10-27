"""Pruebas de la fase 9 (B3): intervenciones sobre la historia de una persona."""

import numpy as np
import pytest

from labor2vec.tokens.documento import INTERVENCIONES, PUNTOS_SALARIO, Eventos, intervenir, opciones_de

FIN = 239


def _ev():
    """Dos relaciones: la 1ª de los meses 200 a 225 (rama 111, tamaño 4); la 2ª desde el 226 (rama 222, tamaño 1)."""
    mes = np.array([200, 201, 225, 226, 230, 235, 239], dtype=np.int16)
    return Eventos(mes_idx=mes,
                   pct_rem=np.array([10, 12, 0, 50, 98, 55, 60], dtype=np.int8),
                   tramo=np.array([4, 4, 4, 1, 1, 1, 1], dtype=np.int8),
                   r34=np.array([111, 111, 111, 222, 222, 222, 222], dtype=np.int16),
                   provi=np.full(7, 2, dtype=np.int8), antig=np.full(7, 3, dtype=np.int8),
                   nueva=np.array([1, 0, 0, 1, 0, 0, 0], dtype=np.int8))


def _copia(ev):
    return {f: getattr(ev, f).copy() for f in ev.__dataclass_fields__}


def test_ninguna_intervencion_modifica_el_original():
    for nombre in INTERVENCIONES:
        ev, antes = _ev(), _copia(_ev())
        intervenir(ev, nombre, FIN)
        assert all(np.array_equal(getattr(ev, f), antes[f]) for f in antes), nombre


def test_sin_ultimo_anio_borra_los_ultimos_12_meses():
    out = intervenir(_ev(), "sin_ultimo_anio", FIN)
    assert out.mes_idx.tolist() == [200, 201, 225, 226] and out.mes_idx.max() <= FIN - 12


def test_sin_cambios_recientes_copia_rama_y_tamano_de_la_relacion_anterior():
    out = intervenir(_ev(), "sin_cambios_recientes", FIN)
    assert out.r34.tolist() == [111] * 7 and out.tramo.tolist() == [4] * 7
    assert out.pct_rem.tolist() == _ev().pct_rem.tolist() and out.mes_idx.tolist() == _ev().mes_idx.tolist()


def test_sin_cambios_recientes_no_hace_nada_con_una_sola_relacion():
    ev = _ev()
    ev.nueva[:] = 0
    ev.nueva[0] = 1
    out = intervenir(ev, "sin_cambios_recientes", FIN)
    assert np.array_equal(out.r34, ev.r34) and np.array_equal(out.tramo, ev.tramo)


def test_salario_sube_y_baja_un_decil_solo_en_el_ultimo_anio():
    base = _ev().pct_rem.astype(int)
    arriba = intervenir(_ev(), "salario_arriba", FIN).pct_rem.astype(int)
    abajo = intervenir(_ev(), "salario_abajo", FIN).pct_rem.astype(int)
    ultimo = _ev().mes_idx > FIN - 12  # eventos de los meses 230, 235 y 239
    assert ultimo.sum() == 3
    assert (arriba[~ultimo] == base[~ultimo]).all() and (abajo[~ultimo] == base[~ultimo]).all()
    assert arriba[ultimo].tolist() == np.clip(base[ultimo] + PUNTOS_SALARIO, 1, 100).tolist() == [100, 65, 70]
    assert abajo[ultimo].tolist() == np.clip(base[ultimo] - PUNTOS_SALARIO, 1, 100).tolist() == [88, 45, 50]


def test_salario_se_acota_y_no_toca_la_remuneracion_cero():
    ev = _ev()
    ev.pct_rem[:] = [0, 0, 0, 0, 0, 97, 0]
    ev.mes_idx[:] = [230, 231, 232, 233, 234, 235, 239]
    assert intervenir(ev, "salario_arriba", FIN).pct_rem.tolist() == [0, 0, 0, 0, 0, 100, 0]
    ev.pct_rem[5] = 4
    assert intervenir(ev, "salario_abajo", FIN).pct_rem.tolist() == [0, 0, 0, 0, 0, 1, 0]


def test_atrasar_2_anios_mueve_las_fechas_y_descarta_lo_que_queda_antes_del_mes_0():
    ev = _ev()
    ev.mes_idx[:] = [5, 20, 24, 25, 30, 235, 239]
    out = intervenir(ev, "atrasar_2_anios", FIN)
    assert out.mes_idx.tolist() == [0, 1, 6, 211, 215]
    assert out.pct_rem.tolist() == ev.pct_rem[2:].tolist()


def test_intervencion_desconocida():
    with pytest.raises(ValueError):
        intervenir(_ev(), "otra", FIN)


def test_opciones_de_lee_la_intervencion():
    assert opciones_de({"max_length": 128, "fin_ventana": 10}).intervencion is None
    assert opciones_de({"max_length": 128, "fin_ventana": 10, "intervencion": "salario_arriba"}).intervencion == "salario_arriba"
