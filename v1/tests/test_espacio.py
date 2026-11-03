"""Pruebas de la fase 9: optimal matching, costos TRATE y estados mensuales (plans/PLAN_fase_9.md)."""

import numpy as np
import polars as pl
from scipy.spatial.distance import squareform

from labor2vec.analisis import secuencias


def om_referencia(a, b, sub, indel):
    """DP de optimal matching en Python puro, para contrastar con la versión de numba."""
    d = np.zeros((len(a) + 1, len(b) + 1))
    d[:, 0] = np.arange(len(a) + 1) * indel
    d[0, :] = np.arange(len(b) + 1) * indel
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d[i, j] = min(d[i - 1, j - 1] + sub[a[i - 1], b[j - 1]], d[i - 1, j] + indel, d[i, j - 1] + indel)
    return d[-1, -1]


def test_om_coincide_con_referencia():
    rng = np.random.default_rng(0)
    seqs = rng.integers(0, 4, size=(12, 15)).astype(np.uint8)
    sub = secuencias.costos_trate(seqs, 4)
    d = squareform(secuencias.distancias_om(seqs, sub, 1.0))
    for i in range(len(seqs)):
        for j in range(len(seqs)):
            assert np.isclose(d[i, j], om_referencia(seqs[i], seqs[j], sub, 1.0), atol=1e-4)
    assert np.allclose(np.diag(d), 0)


def test_om_distancia_cero_entre_iguales_y_simetrica():
    a = np.array([[1, 1, 2, 3, 0, 0], [1, 1, 2, 3, 0, 0], [3, 3, 3, 0, 0, 1]], dtype=np.uint8)
    sub = secuencias.costos_trate(a, 4)
    d = squareform(secuencias.distancias_om(a, sub))
    assert d[0, 1] == 0 and d[0, 2] > 0 and np.allclose(d, d.T)


def test_costos_trate_simetricos_diagonal_cero():
    rng = np.random.default_rng(1)
    seqs = rng.integers(0, 5, size=(200, 30)).astype(np.uint8)
    c = secuencias.costos_trate(seqs, 5)
    assert np.allclose(c, c.T) and np.allclose(np.diag(c), 0)
    assert (c >= 0).all() and (c <= 2).all()


def test_estados_mensuales(tmp_path):
    # persona 1: dos relaciones en el mes 3 (gana la de mayor remuneración); persona 2: sin registros en la ventana
    panel = pl.DataFrame({
        "id_trabajador": [1, 1, 1, 1], "id_relacion": [10, 10, 11, 10], "mes_idx": [2, 3, 3, 4],
        "rem_tot": [100.0, 100.0, 500.0, 100.0], "pct_rem_mes": [10.0, 10.0, 90.0, 50.0], "tramo_empleo": [4, 4, 1, 2]})
    ruta = tmp_path / "panel.parquet"
    panel.write_parquet(ruta)
    s = secuencias.estados_mensuales(np.array([1, 2]), fin=5, meses=6, panel=ruta)
    assert s.shape == (2, 6)
    # estado = 1 + 3·salario + tamaño: (bajo, chica) = 1; (alto, grande) = 1 + 6 + 2 = 9; (medio, mediana) = 1 + 3 + 1 = 5
    assert s[0].tolist() == [0, 0, 1, 9, 5, 0]
    assert (s[1] == 0).all()


def test_sonda_continua_recupera_informacion_lineal_y_no_inventa():
    from labor2vec.analisis import espacio

    rng = np.random.default_rng(0)
    x = rng.normal(size=(3000, 20))
    y_lineal = x[:, 0] * 2 - x[:, 3] + 0.1 * rng.normal(size=3000)
    y_ruido = rng.normal(size=3000)
    r = espacio.sonda_continua(x[:1500], y_lineal[:1500], x[1500:], y_lineal[1500:], n_boot=30)
    assert r["valor"] > 0.95 and r["lo"] <= r["valor"] <= r["hi"]
    r0 = espacio.sonda_continua(x[:1500], y_ruido[:1500], x[1500:], y_ruido[1500:], n_boot=30)
    assert abs(r0["valor"]) < 0.05


def test_sonda_categorica_sobre_el_azar_solo_con_informacion():
    from labor2vec.analisis import espacio

    rng = np.random.default_rng(1)
    x = rng.normal(size=(4000, 10))
    y = (x[:, 0] + 0.3 * rng.normal(size=4000) > 0).astype(int) + (x[:, 1] > 1).astype(int)
    r = espacio.sonda_categorica(x[:2000], y[:2000], x[2000:], y[2000:], n_boot=30)
    assert r["valor"] > 0.6 > r["azar"] + 0.2
    y_azar = rng.integers(0, 3, 4000)
    r0 = espacio.sonda_categorica(x[:2000], y_azar[:2000], x[2000:], y_azar[2000:], n_boot=30)
    assert abs(r0["valor"] - 1 / 3) < 0.05


def test_razon_de_correlacion():
    from labor2vec.analisis import espacio

    g = np.repeat([0, 1, 2], 100)
    assert espacio.razon_de_correlacion(g.astype(float), g) == 1.0
    x = np.random.default_rng(0).normal(size=300)
    assert espacio.razon_de_correlacion(x, g) < 0.05
    assert espacio.razon_de_correlacion(x, np.full(300, -1)) == 0.0  # sin grupos válidos


def test_sonda_continua_con_variable_constante_da_nan():
    from labor2vec.analisis import espacio

    x = np.random.default_rng(0).normal(size=(200, 5))
    r = espacio.sonda_continua(x[:100], np.zeros(100), x[100:], np.zeros(100), n_boot=5)
    assert np.isnan(r["valor"])


def test_submatriz_coincide_con_squareform():
    from scipy.spatial.distance import pdist

    from labor2vec.analisis import espacio

    x = np.random.default_rng(0).normal(size=(40, 3))
    d = pdist(x).astype(np.float32)
    completa = squareform(d)
    idx = np.array([30, 2, 17, 5, 39])
    sub = espacio.submatriz(d, 40, idx)
    esperado = completa[np.ix_(np.sort(idx), np.sort(idx))]
    assert np.allclose(sub, esperado, atol=1e-6) and np.allclose(np.diag(sub), 0)


def test_cortar_y_normalizar():
    from scipy.cluster.hierarchy import linkage

    from labor2vec.analisis import espacio

    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(size=(30, 4)) + 8, rng.normal(size=(30, 4)) - 8])
    lab = espacio.cortar(linkage(x, "ward"), 2)
    assert set(lab) == {0, 1} and len(set(lab[:30])) == 1 and len(set(lab[30:])) == 1 and lab[0] != lab[30]
    assert np.allclose(np.linalg.norm(espacio._normalizar(x), axis=1), 1)


def test_conteos_lee_la_cache_y_seccion_conteos_compara_ocho_espacios(tmp_path, monkeypatch):
    from labor2vec.analisis import espacio

    ids = np.arange(100)
    c = np.random.default_rng(0).integers(0, 5, size=(100, 6)).astype(np.float32)
    monkeypatch.setattr(espacio, "TAREAS", tmp_path)
    np.savez(tmp_path / f"conteo_tokens_val_v{espacio.FIN_VENTANA}.npz", id=ids, c=c)
    i, x = espacio.conteos("val")  # no toca el corpus: lee la caché
    assert (i == ids).all() and (x == c).all()

    np.savez(tmp_path / f"conteo_tokens_test_v{espacio.FIN_VENTANA}.npz", id=ids, c=c)
    prop = c / np.maximum(c.sum(1, keepdims=True), 1)
    y = np.log1p(c.sum(1))  # una variable que sólo los conteos (no las proporciones) pueden dar
    anual = np.random.default_rng(1).normal(size=(100, 8)).astype(np.float32)
    monkeypatch.setattr(espacio, "por_anio", lambda s: (ids, anual))
    monkeypatch.setattr(espacio, "proporciones", lambda s: (ids, prop))
    monkeypatch.setattr(espacio, "espacios_de", lambda e: {"pre": {s: (ids, prop[:, :2]) for s in ("val", "test")}})
    monkeypatch.setattr(espacio, "objetivos", lambda df, i, s: ({}, {"dinámica: largo": y}))
    monkeypatch.setattr(espacio, "dinamica_estandarizada", lambda a, b: (a, b))
    monkeypatch.setattr(espacio.pl, "read_parquet", lambda *a, **k: None)
    texto = espacio.seccion_conteos(tmp_path, n_boot=5)
    filas = {f.split("|")[1].strip(): float(f.split("|")[2]) for f in texto.splitlines() if f.startswith("| ") and "---" not in f
             and not f.startswith("| espacio")}
    assert len(filas) == 8
    assert filas["conteos absolutos"] > 0.99 or filas["log(1 + conteos) + tamaño"] > 0.99  # el largo se lee de los conteos
    assert filas["proporciones"] < 0.5  # y no de las proporciones


def test_por_anio_cuenta_meses_inicios_y_cambios_por_ano_calendario():
    from labor2vec.tareas import tiempo

    # relación 1: meses 10 a 25 (1996-11 a 1998-02); relación 2: de 30 a 40, con rama distinta; tamaño de la 2 cambia en el 36
    mes = np.array(list(range(10, 26)) + list(range(30, 41)))
    rel = np.array([1] * 16 + [2] * 11)
    nueva = np.array([True] + [False] * 15 + [True] + [False] * 10)
    rama = np.array([5] * 16 + [7] * 11)
    tramo = np.array([2] * 16 + [2] * 6 + [3] * 5)
    x = tiempo.por_anio(mes, np.ones(len(mes)), tramo, rama, rel, nueva).reshape(4, tiempo.ANIOS)
    esperado_meses = np.bincount(mes // 12, minlength=tiempo.ANIOS)
    assert x[0].tolist() == esperado_meses.tolist()
    assert x[1].sum() == 1 and x[1, 30 // 12] == 1  # sólo el inicio de la relación 2 (el de la 1 es el primer mes)
    assert x[2].sum() == 1 and x[2, 36 // 12] == 1  # un cambio de tamaño en el mes 36
    assert x[3].sum() == 1 and x[3, 30 // 12] == 1  # un cambio de rama al empezar la relación 2
    assert (tiempo.por_anio(mes[:0], mes[:0], mes[:0], mes[:0], mes[:0], mes[:0].astype(bool)) == 0).all()


def test_por_anio_ignora_eventos_posteriores_al_corte_en_todos_los_arreglos():
    from labor2vec.tareas import tiempo

    mes = np.array([5, 6, 7, 230, 235, 238, 241, 245, 250])  # los tres últimos pasan el corte (239) y están en 2015+
    rem = np.arange(1, 10, dtype=float)
    rel = np.array([1, 1, 1, 2, 2, 2, 2, 2, 2])
    nueva = np.array([True, False, False, True, False, False, False, False, False])
    x = tiempo.por_anio(mes, rem, np.full(9, 2), np.full(9, 5), rel, nueva, 239)  # no debe fallar
    assert x.reshape(4, tiempo.ANIOS)[0].sum() == 6  # sólo los meses hasta el corte
