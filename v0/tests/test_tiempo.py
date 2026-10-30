"""Pruebas de las variables de tiempo de las líneas de base tabulares (fase 8b)."""

import numpy as np

from labor2vec.tareas import tiempo
from labor2vec.tareas.tiempo import NOMBRES


def _eventos(*tramos):
    """tramos: (mes_ini, mes_fin, relación, tramo, rama, remuneración); un evento por mes, `nueva` en el primero de
    cada relación."""
    mes, rel, tam, rama, rem, nueva, vistas = [], [], [], [], [], [], set()
    for a, b, r, t, ra, p in tramos:
        for m in range(a, b + 1):
            mes.append(m), rel.append(r), tam.append(t), rama.append(ra), rem.append(p)
            nueva.append(r not in vistas and m == a)
        vistas.add(r)
    o = np.argsort(mes, kind="stable")
    arr = lambda x, d: np.array(x, dtype=d)[o]  # noqa: E731
    return dict(mes=arr(mes, np.int16), rem=arr(rem, np.int8), tramo=arr(tam, np.int8), rama=arr(rama, np.int16),
                rel=arr(rel, np.int16), nueva=arr(nueva, bool))


def _v(ev, **kw):
    return dict(zip(NOMBRES, tiempo.variables(**ev, **kw), strict=True))


def test_cambio_de_empresa_con_historia_continua():
    v = _v(_eventos((100, 219, 1, 1, 10, 40), (220, 239, 2, 2, 20, 60)))
    assert v["antiguedad"] == 20  # principal de 2015 = la relación 2, desde el mes 220
    assert v["meses_desde_cambio_empresa"] == 19  # el único cambio fue el inicio de la relación 2
    assert (v["cambios_empresa_12"], v["cambios_empresa_24"], v["cambios_empresa_60"]) == (0, 1, 1)
    assert v["meses_desde_cambio_tamano"] == 19 and v["cambios_tamano_60"] == 0  # sin cambio dentro de la principal
    assert v["huecos_24"] == 0 and v["huecos_60"] == 0 and v["meses_desde_ultimo_registro"] == 0
    assert (v["dif_salario_12"], v["dif_salario_36"], v["hay_dif_12"], v["hay_dif_36"]) == (0, 20, 1, 1)


def test_cambios_de_tamano_y_de_rama():
    v = _v(_eventos((100, 229, 1, 1, 10, 50), (230, 239, 1, 2, 10, 50)))  # el tamaño cambia dentro de la relación
    assert v["meses_desde_cambio_tamano"] == 9 and v["cambios_tamano_12"] == 1
    assert v["cambios_empresa_60"] == 0  # el primer inicio de la historia no cuenta como cambio
    assert v["meses_desde_cambio_empresa"] == 139  # sin cambios: desde el primer mes (239 - 100)
    assert v["cambios_rama_60"] == 0 and v["meses_desde_cambio_rama"] == 139

    # la rama sólo cambia entre relaciones: la 2 (misma rama que la 1) no cuenta, la 3 sí
    w = _v(_eventos((100, 149, 1, 1, 10, 50), (150, 199, 2, 1, 10, 50), (200, 239, 3, 1, 11, 50)))
    assert w["meses_desde_cambio_rama"] == 39 and w["cambios_rama_60"] == 1 and w["cambios_rama_12"] == 0
    assert w["cambios_empresa_60"] == 1 and w["meses_desde_cambio_empresa"] == 39


def test_huecos_cuentan_sólo_desde_que_la_persona_entra():
    con_hueco = _v(_eventos((100, 150, 1, 1, 10, 50), (200, 239, 2, 1, 10, 50)))
    assert con_hueco["huecos_24"] == 0 and con_hueco["huecos_60"] == 20  # ventana 180-239, registrados 200-239
    recien_llegado = _v(_eventos((200, 239, 1, 1, 10, 50)))
    assert recien_llegado["huecos_60"] == 0  # no se cuentan los meses anteriores a su primer registro


def test_ultimo_registro_y_salario_sin_dato():
    v = _v(_eventos((220, 230, 1, 1, 10, 50)))
    assert v["meses_desde_ultimo_registro"] == 9 and v["hay_dif_12"] == 1
    assert v["hay_dif_36"] == 0 and v["dif_salario_36"] == 0  # no hay valor 36 meses antes de 239


def test_no_mira_despues_del_corte():
    base = _eventos((100, 239, 1, 1, 10, 50))
    con_futuro = _eventos((100, 239, 1, 1, 10, 50), (240, 260, 2, 3, 99, 99))
    assert np.array_equal(tiempo.variables(**base), tiempo.variables(**con_futuro))


def test_conjuntos():
    todos, sin = tiempo.indices("completo"), tiempo.indices("sin_ultimo_valor")
    assert len(todos) == len(NOMBRES) == 20
    quitados = {NOMBRES[k] for k in todos if k not in sin}
    assert quitados == {"antiguedad"} | {n for n in NOMBRES if "rama" in n} and len(sin) == 20 - 5


def test_transformar_usa_estadisticas_de_train():
    x = np.abs(np.random.default_rng(0).normal(5, 2, size=(200, len(NOMBRES))))
    z, media, desvio = tiempo.transformar(x[:150])
    assert abs(z.mean()) < 1e-3
    z2, _, _ = tiempo.transformar(x[150:], media, desvio)
    assert z2.shape == (50, len(NOMBRES)) and not np.allclose(z2.mean(0), 0, atol=1e-3)
