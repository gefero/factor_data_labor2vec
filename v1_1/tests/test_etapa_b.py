"""Pruebas de la fase 9, etapa B: medidas de estabilidad, movimiento e intervenciones, y las secciones del informe."""

from pathlib import Path

import numpy as np
import pytest

from labor2vec.analisis import etapa_b as eb
from labor2vec.analisis import personas
from labor2vec.tareas.metricas import aul

RAIZ = Path(__file__).resolve().parents[1]
REAL = RAIZ / "runs/fase7/pre/personas.npz"


# ---- medidas puras ---------------------------------------------------------------------------------------------------
def test_cka_idéntico_rotado_y_escalado_da_1_y_el_azar_da_poco():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 12))
    q, _ = np.linalg.qr(rng.normal(size=(12, 12)))
    assert eb.cka_lineal(x, x) == pytest.approx(1.0)
    assert eb.cka_lineal(x, 3.0 * (x @ q)) == pytest.approx(1.0)
    assert eb.cka_lineal(x, rng.normal(size=(500, 12))) < 0.1


def test_solapamiento_de_vecinos_identico_y_azar():
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=(400, 8)), rng.normal(size=(400, 8))
    obs, azar = eb.solapamiento_vecinos(x, x, k=10, n=400)
    assert obs == 1.0 and azar == pytest.approx(10 / 399)
    obs_y, _ = eb.solapamiento_vecinos(x, y, k=10, n=400)
    assert obs_y < 0.15


def test_alinear_a_reordena_y_marca_los_que_faltan():
    ids = np.array([5, 3, 9, 1])
    x = np.array([[50], [30], [90], [10]])
    ok, (y,) = eb.alinear_a(np.array([1, 2, 3, 9]), ids, x)
    assert ok.tolist() == [True, False, True, True] and y.ravel().tolist() == [10, 30, 90]


def test_matriz_transicion_filas_suman_uno():
    antes = np.array([0, 0, 0, 1, 1, 2])
    despues = np.array([0, 0, 1, 1, 1, 0])
    t = eb.matriz_transicion(antes, despues, 3)
    assert t[0].tolist() == pytest.approx([2 / 3, 1 / 3, 0]) and t[1].tolist() == [0, 1, 0] and t[2].tolist() == [1, 0, 0]
    assert np.allclose(t.sum(1), 1)


def test_resumen_logit_cambio_medio_y_aul():
    y = np.array([0, 0, 1, 1])
    base = np.array([-1.0, 0.0, 0.5, 1.0])
    nuevo = np.array([-1.0, 0.5, 0.5, 2.0])
    r = eb.resumen_logit(base, nuevo, y, np.array([False, True, False, True]))
    assert r["n_afectadas"] == 2 and r["media"] == pytest.approx(0.75) and r["sube"] == 1.0
    assert r["aul_antes"] == pytest.approx(aul(y, base)) and r["aul_despues"] == pytest.approx(aul(y, nuevo))
    assert r["aul_despues"] <= r["aul_antes"]  # el empate que se introdujo no mejora la AUL


def test_distancia_coseno_filas_y_tipologia():
    a = np.array([[1.0, 0], [0, 1.0]])
    assert eb.distancia_coseno_filas(a, a).tolist() == [0, 0]
    assert eb.distancia_coseno_filas(a, a[::-1]).tolist() == pytest.approx([1.0, 1.0])
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(size=(40, 5)) + 6, rng.normal(size=(40, 5)) - 6])
    lab = eb.tipologia(x, 2)
    assert len(set(lab[:40])) == 1 and len(set(lab[40:])) == 1 and lab[0] != lab[40]


def test_cargar_devuelve_none_si_falta(tmp_path):
    assert eb.cargar(tmp_path / "no_existe") is None


# ---- secciones, con carpetas simuladas y ids reales -------------------------------------------------------------------
@pytest.fixture(scope="module")
def mundo(tmp_path_factory):
    if not REAL.exists() or not personas.CONCEPTOS.exists():
        pytest.skip("faltan los vectores de la fase 7 o los conceptos de personas")
    z = np.load(REAL)
    rng = np.random.default_rng(0)
    n, d = 700, 16
    base = {s: {"id": z[f"{s}_id"][:n], "y": z[f"{s}_y"][:n], "vec": rng.normal(size=(n, d)).astype(np.float32)} for s in ("val", "test")}
    e7, e9 = tmp_path_factory.mktemp("fase7"), tmp_path_factory.mktemp("fase9")

    def guardar(carpeta, nombre, datos, splits=("val", "test")):
        (carpeta / nombre).mkdir(parents=True, exist_ok=True)
        np.savez(carpeta / nombre / "personas.npz", **{f"{s}_{k}": v for s in splits for k, v in datos[s].items()})

    def con_logit(datos):
        return {s: {**v, "logit": (v["vec"][:, 0] + 0.1 * rng.normal(size=n)).astype(np.float32)} for s, v in datos.items()}

    sal = {}
    for fam in ("pre", "t1", "t3"):
        b = con_logit(base) if fam != "pre" else base
        guardar(e7, fam, b)
        sal[fam] = b
        for sem in (1, 2):
            ruido = {s: {**v, "vec": (v["vec"] + 0.3 * rng.normal(size=v["vec"].shape)).astype(np.float32)} for s, v in b.items()}
            guardar(e9, f"{fam}_s{sem}", ruido)
    # cortes del pre: sólo test
    for corte in (179, 203, 227):
        v = sal["pre"]["test"]
        k = slice(None) if corte != 179 else slice(None, 600)  # en un corte temprano faltan las personas sin eventos todavía
        guardar(e9, f"pre_f{corte}", {"test": {"id": v["id"][k][::-1], "vec": (v["vec"][k][::-1] + 0.5 * rng.normal(size=v["vec"][k].shape)).astype(np.float32)}},
                splits=("test",))
    # intervenciones: se cambian exactamente las primeras 100 personas
    for fam in ("t1", "t3", "pre"):
        for nombre in eb.INTERVENCIONES:
            v = {k: a.copy() for k, a in sal[fam]["test"].items()}
            v["vec"][:100] += 1.0
            if "logit" in v:
                v["logit"][:100] += 0.5
            guardar(e9, f"{fam}_{nombre}", {"test": v}, splits=("test",))
    return {"e7": e7, "e9": e9, "n": n}


@pytest.fixture
def sondas_falsas(monkeypatch):
    claves = eb.CLAVE_CAT + eb.CLAVE_CONT
    monkeypatch.setattr(eb, "sondear", lambda val, test, df, n_boot=30: {c: 0.5 for c in claves})


def test_seccion_semillas(mundo, sondas_falsas):
    t = eb.seccion_semillas(mundo["e7"], mundo["e9"], k=4)
    for fam in ("pre", "t1", "t3"):
        assert f"### {fam}\n" in t
    assert "| 0 y 1 |" in t and "| 1 y 2 |" in t and "Rango de las sondas" in t


def test_seccion_cortes(mundo):
    t = eb.seccion_cortes(mundo["e7"], mundo["e9"], k=4)
    assert "### Corte 2010-12" in t and "### Corte 2014-12" in t and "Transición entre tipos" in t and "Permanencia" in t
    assert "2010-12: 600" in t  # sólo las personas presentes en el corte


def test_seccion_cortes_sin_datos_lo_dice(tmp_path):
    if not REAL.exists():
        pytest.skip("faltan los vectores de la fase 7")
    assert "Sin datos" in eb.seccion_cortes(REAL.parents[1], tmp_path)


def test_seccion_intervenciones_cuenta_las_afectadas(mundo):
    t = eb.seccion_intervenciones(mundo["e7"], mundo["e9"], k=4)
    n = mundo["n"]
    esperado = f"100 ({round(100 * 100 / n)} %)"
    for nombre in eb.INTERVENCIONES:
        assert nombre in t
    assert t.count(esperado) >= 3 * len(eb.INTERVENCIONES)
    assert "+0.500" in t  # el cambio de logit que se puso a mano (sólo t1 y t3)


def test_secciones_de_controles_y_extras(mundo, sondas_falsas):
    assert "Sin datos" in eb.seccion_controles(mundo["e7"], mundo["e9"])
    assert "Sin datos" in eb.seccion_extras(mundo["e7"], mundo["e9"])


def test_controles_y_extras_sondean_una_vez_por_corrida(mundo, monkeypatch):
    """Antes se llamaba a `sondear` una vez por columna de la tabla (13 veces el mismo trabajo)."""
    import shutil

    claves, llamadas = eb.CLAVE_CAT + eb.CLAVE_CONT, []
    monkeypatch.setattr(eb, "sondear", lambda val, test, df, n_boot=30: llamadas.append(1) or {c: 0.5 for c in claves})
    for destino, origen in (("pre_ctrl_tiempo", "pre_s1"), ("pre_ctrl_orden", "pre_s2"), ("pre_sinuv", "pre_s1"), ("pre_azar", "pre_s2")):
        shutil.copytree(mundo["e9"] / origen, mundo["e9"] / destino, dirs_exist_ok=True)
    assert "| fechas permutadas |" in eb.seccion_controles(mundo["e7"], mundo["e9"])
    assert len(llamadas) == 3  # pre, fechas permutadas, orden permutado
    llamadas.clear()
    assert "| pesos sin entrenar |" in eb.seccion_extras(mundo["e7"], mundo["e9"])
    assert len(llamadas) == 3


def test_tipologia_con_muestra_recupera_grupos_y_asigna_a_todos():
    from sklearn.metrics import adjusted_rand_score

    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(size=(60, 5)) + 8 * np.eye(5)[j] for j in range(3)])  # tres direcciones distintas
    verdad = np.repeat([0, 1, 2], 60)
    perm = rng.permutation(len(x))
    a, b = np.sort(perm[:90]), np.sort(perm[90:])
    ta, tb = eb.tipologia_con_muestra(x, a, 3), eb.tipologia_con_muestra(x, b, 3)
    assert len(ta) == len(x) == len(tb)  # asigna a todas, no sólo a la muestra
    assert adjusted_rand_score(verdad, ta) > 0.9 and adjusted_rand_score(ta, tb) > 0.9
    ruido = rng.normal(size=(180, 5))  # sin grupos reales las dos mitades casi no coinciden
    assert adjusted_rand_score(eb.tipologia_con_muestra(ruido, a, 3), eb.tipologia_con_muestra(ruido, b, 3)) < 0.5
