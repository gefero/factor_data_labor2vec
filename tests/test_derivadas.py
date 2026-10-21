"""Pruebas de la fase 6b: etiquetas de T2, T3 y T4."""

import numpy as np
import polars as pl
import pytest

from labor2vec.tareas import derivadas

ANO15 = list(range(228, 240))
ANO19 = list(range(276, 288))


def _filas(t, rel, meses, letra, tramo, rem, pct):
    return [(t, rel, m, letra, tramo, rem, pct) for m in meses]


@pytest.fixture
def datos(tmp_path):
    f = []
    # 1: sin cambios, salario estable (train)
    f += _filas(1, 1, ANO15, 5, 3, 100.0, 50) + _filas(1, 1, ANO19, 5, 3, 100.0, 50)
    # 2: empate de meses en 2015 (6 y 6): gana la relación de mayor remuneración (letra 7); en 2019 cambia a letra 9 y
    #    pasa a una empresa más grande (tramo 3 -> 1)
    f += _filas(2, 1, ANO15[:6], 3, 4, 10.0, 20) + _filas(2, 2, ANO15[6:], 7, 3, 90.0, 60)
    f += _filas(2, 1, ANO19, 9, 1, 100.0, 70)
    # 3: tramo 0 («sin definir») en 2019: queda en T2 y T3 pero no en T4
    f += _filas(3, 1, ANO15, 5, 2, 100.0, 40) + _filas(3, 1, ANO19, 5, 0, 100.0, 45)
    # 4: sale del empleo en 2019 (no entra a ninguna tarea)
    f += _filas(4, 1, ANO15, 5, 2, 100.0, 40)
    # 5: remuneración cero todo 2019: no entra a T2
    f += _filas(5, 1, ANO15, 5, 2, 100.0, 30) + _filas(5, 1, ANO19, 5, 2, 0.0, 0)
    # 6: pasa a una empresa más chica (2 -> 4) y letra igual (val)
    f += _filas(6, 1, ANO15, 4, 2, 100.0, 80) + _filas(6, 1, ANO19, 4, 4, 100.0, 85)
    # 7: percentil 2019 muy alto (test)
    f += _filas(7, 1, ANO15, 4, 2, 100.0, 90) + _filas(7, 1, ANO19, 4, 2, 100.0, 99)
    panel = pl.DataFrame(f, schema=["id_trabajador", "id_relacion", "mes_idx", "letra", "tramo_empleo", "rem_tot",
                                    "pct_rem_mes"], orient="row")
    ids = [1, 2, 3, 4, 5, 6, 7]
    t1 = pl.DataFrame({"id_trabajador": ids, "split": ["train", "train", "train", "train", "train", "val", "test"],
                       "sexo": [1, 2, 1, 2, 1, 2, 1], "fnacim": [1980] * 7, "edad_corte": [35] * 7,
                       "y_2019": [False, False, False, True, False, False, False],
                       "y_2021": [False] * 7})
    panel.write_parquet(tmp_path / "panel.parquet")
    t1.write_parquet(tmp_path / "t1.parquet")
    return tmp_path


def test_poblacion_y_estado(datos):
    estado = pl.DataFrame(derivadas.conectar().execute(derivadas.sql_estado(datos / "panel.parquet",
                                                                            datos / "t1.parquet")).pl())
    assert estado["id_trabajador"].to_list() == [1, 2, 3, 5, 6, 7]  # el 4 salió en 2019
    f = estado.filter(pl.col("id_trabajador") == 2).row(0, named=True)
    assert (f["letra2015"], f["tramo2015"]) == (7, 3)  # empate de meses: gana la de mayor remuneración
    assert (f["letra2019"], f["tramo2019"]) == (9, 1)
    assert f["pct2015"] == pytest.approx(40.0)  # promedio de las 12 filas: (6*20 + 6*60) / 12


def test_tablas_y_exclusiones(datos):
    out = derivadas.construir(datos / "panel.parquet", datos / "t1.parquet", datos / "salida")
    assert out["t3"]["id_trabajador"].to_list() == [1, 2, 3, 5, 6, 7]
    assert out["t3"].sort("id_trabajador")["y"].to_list() == [0, 1, 0, 0, 0, 0]
    # T4: el 3 tiene tramo 0 en 2019 y no entra; 0 = más chica, 1 = igual, 2 = más grande
    t4 = out["t4"].sort("id_trabajador")
    assert t4["id_trabajador"].to_list() == [1, 2, 5, 6, 7]
    assert t4["y"].to_list() == [1, 2, 1, 0, 1]
    # T2: el 5 (remuneración cero en 2019) no entra
    assert out["t2"].sort("id_trabajador")["id_trabajador"].to_list() == [1, 2, 3, 6, 7]
    for n in ("t2", "t3", "t4"):
        assert (datos / "salida" / f"{n}.parquet").exists()
        assert {"id_trabajador", "split", "sexo", "fnacim", "edad_corte", "y"} <= set(out[n].columns)


def test_cortes_de_quintiles_solo_con_train(datos):
    estado = derivadas.conectar().execute(derivadas.sql_estado(datos / "panel.parquet", datos / "t1.parquet")).pl()
    cortes = derivadas.cortes_quintiles(estado)
    train = estado.filter(pl.col("split") == "train")["pct2019"].drop_nulls().to_numpy()
    assert np.allclose(cortes, np.quantile(train, derivadas.QUINTILES))
    # cambiar un valor de test no mueve los cortes
    mod = estado.with_columns(pl.when(pl.col("split") == "test").then(1.0).otherwise(pl.col("pct2019")).alias("pct2019"))
    assert np.allclose(derivadas.cortes_quintiles(mod), cortes)
    q = derivadas.quintil(np.array([0.0, cortes[0], cortes[1] - 1e-9, cortes[3], 100.0]), cortes)
    assert q.tolist() == [0, 1, 1, 4, 4]
    assert derivadas.resumen(derivadas.tablas(estado)).startswith("t2")
