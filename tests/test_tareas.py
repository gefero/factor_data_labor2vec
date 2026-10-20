"""Pruebas de la fase 6: cohorte y etiquetas de T1."""

import polars as pl

from labor2vec.tareas import t1


def _panel(tmp_path):
    """Trabajadores con casos de borde. Cada uno: (id, fnacim, sexo, meses con registro)."""
    ano2015 = list(range(228, 240))
    casos = [
        (1, 1980, 1, ano2015 + [276]),                 # cohorte; sigue en 2019, no está en 2021
        (2, 1980, 2, ano2015 + [300]),                 # cohorte; sale en 2019, vuelve en 2021
        (3, 1980, 1, ano2015[:11]),                    # 11 meses en 2015: afuera
        (4, 1990, 1, ano2015),                         # 25 años justos: adentro (y=1 en ambos)
        (5, 1991, 1, ano2015),                         # 24 años: afuera
        (6, 1960, 2, ano2015 + [287]),                 # 55 años justos: adentro; 2019 sí (mes 287 = 2019-12)
        (7, 1959, 2, ano2015),                         # 56 años: afuera
        (8, 1980, 1, ano2015 + [275]),                 # 2018-12: no cuenta como 2019
        (9, 1980, 2, ano2015),                         # sin split: excluido por el join
    ]
    filas = [(t, f, s, m) for t, f, s, ms in casos for m in ms]
    panel = pl.DataFrame(filas, schema=["id_trabajador", "fnacim", "sexo", "mes_idx"], orient="row")
    splits = pl.DataFrame({"id_trabajador": list(range(1, 9)),
                           "split": ["train", "val", "test", "train", "train", "val", "test", "train"]})
    panel.write_parquet(tmp_path / "panel.parquet")
    splits.write_parquet(tmp_path / "splits.parquet")
    return tmp_path / "panel.parquet", tmp_path / "splits.parquet"


def test_ventanas():
    assert t1._ventana(2015) == (228, 239)
    assert t1._ventana(2019) == (276, 287)
    assert t1._ventana(2021) == (300, 311)


def test_cohorte_y_etiquetas(tmp_path):
    panel, splits = _panel(tmp_path)
    df = t1.construir(panel, splits, tmp_path / "t1.parquet").sort("id_trabajador")
    assert df["id_trabajador"].to_list() == [1, 2, 4, 6, 8]
    assert df["edad_corte"].to_list() == [35, 35, 25, 55, 35]
    assert df["y_2019"].to_list() == [False, True, True, False, True]
    assert df["y_2021"].to_list() == [True, False, True, True, True]
    assert df["split"].to_list() == ["train", "val", "train", "val", "train"]
    assert pl.read_parquet(tmp_path / "t1.parquet").height == 5


def test_resumen(tmp_path):
    panel, splits = _panel(tmp_path)
    texto = t1.resumen(t1.construir(panel, splits, tmp_path / "t1.parquet"))
    assert "total" in texto and "35-44" in texto


# ---- métricas ----------------------------------------------------------------------------------------------------
import numpy as np
import pytest
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

from labor2vec.tareas import metricas


def _datos(n=2000, semilla=0):
    rng = np.random.default_rng(semilla)
    y = rng.random(n) < 0.2
    s = np.round(rng.normal(size=n) + 1.2 * y, 1)  # con empates a propósito
    return y, s


def test_auc_aul_y_pr_coinciden_con_sklearn():
    y, s = _datos()
    assert metricas.auc(y, s) == pytest.approx(roc_auc_score(y, s))
    assert metricas.pr_auc(y, s) == pytest.approx(average_precision_score(y, s))
    c = y.mean()
    assert metricas.aul(y, s) == pytest.approx(c / 2 + (1 - c) * roc_auc_score(y, s))


def test_aul_perfecto_y_azar():
    y = np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 0])
    c = y.mean()
    assert metricas.aul(y, y.astype(float)) == pytest.approx(c / 2 + (1 - c))
    assert metricas.aul(y, np.zeros(len(y))) == pytest.approx(0.5)  # puntajes constantes: AUL = 0,5


def test_cmcc_con_alfa_cero_es_el_mcc_comun():
    y, s = _datos(semilla=1)
    p = 1 / (1 + np.exp(-s))
    r = metricas.corregidas(y, p)
    assert r["c_mcc"] == pytest.approx(matthews_corrcoef(y, p >= 0.5))
    assert r["c_mcc"] == pytest.approx(r["mcc"])


def test_cmcc_corregido_cambia_con_alfa():
    y, s = _datos(semilla=2)
    p = 1 / (1 + np.exp(-s))
    assert metricas.corregidas(y, p, alfa=0.05)["c_mcc"] != pytest.approx(metricas.corregidas(y, p)["c_mcc"])


def test_correccion_de_prevalencia():
    assert metricas.corregir_prevalencia(np.array([0.5]), 0.16)[0] == pytest.approx(0.16)
    p = np.array([0.1, 0.5, 0.9])
    q = metricas.corregir_prevalencia(p, 0.16)
    assert (np.diff(q) > 0).all() and (q < p + 1e-12).all()


def test_bootstrap_contiene_el_valor_puntual():
    y, s = _datos(n=800)
    p = 1 / (1 + np.exp(-s))
    r = metricas.bootstrap(y, [p, p * 0.9 + 0.01], n=100)
    for media, lo, hi in r.values():
        assert lo <= media <= hi or media == pytest.approx(lo) or media == pytest.approx(hi)
    assert r["auc"][2] - r["auc"][1] < 0.1


def test_umbral_optimo_y_calibracion():
    y, s = _datos()
    p = 1 / (1 + np.exp(-s))
    u = metricas.umbral_optimo(y, p)
    assert metricas.corregidas(y, p, u)["mcc"] >= metricas.corregidas(y, p, 0.99)["mcc"]
    cal = metricas.calibracion(y, p, 5)
    assert cal.shape == (5, 3) and cal[:, 2].sum() == len(y)
    assert (np.diff(cal[:, 1]) > -0.1).all()
