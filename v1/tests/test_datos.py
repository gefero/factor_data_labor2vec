"""Invariantes de los datos procesados de la fase 1. Se saltean si data/processed no existe."""

import pytest

from labor2vec.utiles.rutas import PANEL, RELACIONES, SPLITS, TRABAJADORES

pytestmark = pytest.mark.skipif(not PANEL.exists(), reason="faltan los datos procesados (python -m labor2vec.datos)")


@pytest.fixture(scope="module")
def con():
    from labor2vec.utiles.duck import conectar

    return conectar()


def uno(con, sql):
    return con.execute(sql).fetchone()[0]


def test_clave_unica(con):
    assert uno(con, f"""SELECT count(*) FROM (SELECT 1 FROM '{PANEL}'
                        GROUP BY id_trabajador, id_relacion, tiempo HAVING count(*) > 1)""") == 0


def test_atributos_fijos_por_relacion(con):
    assert uno(con, f"""SELECT count(*) FROM (SELECT 1 FROM '{PANEL}' GROUP BY id_trabajador, id_relacion
                        HAVING count(DISTINCT provi) > 1 OR count(DISTINCT r34) > 1
                            OR count(DISTINCT tr_anio_ini) > 1)""") == 0


def test_atributos_fijos_por_trabajador(con):
    assert uno(con, f"""SELECT count(*) FROM (SELECT 1 FROM '{PANEL}' GROUP BY id_trabajador
                        HAVING count(DISTINCT sexo) > 1 OR count(DISTINCT fnacim) > 1)""") == 0


def test_jerarquia_de_ramas(con):
    assert uno(con, f"SELECT count(*) FROM '{PANEL}' WHERE r34 // 100 <> r32") == 0
    assert uno(con, f"""SELECT count(*) FROM (SELECT r32 FROM '{PANEL}' WHERE r32 IS NOT NULL
                        GROUP BY r32 HAVING count(DISTINCT letra) > 1)""") == 0


def test_sin_codigos_cero(con):
    for v in ("provi", "letra", "r32", "r34", "tramo_empleo", "sexo"):
        assert uno(con, f"SELECT count(*) FROM '{PANEL}' WHERE {v} = 0") == 0, v


def test_splits_disjuntos_y_completos(con):
    assert uno(con, f"SELECT count(*) - count(DISTINCT id_trabajador) FROM '{SPLITS}'") == 0
    elegibles = uno(con, f"SELECT count(*) FROM '{TRABAJADORES}' WHERE motivo_exclusion IS NULL")
    assert uno(con, f"SELECT count(*) FROM '{SPLITS}'") == elegibles
    assert uno(con, f"""SELECT count(*) FROM '{SPLITS}' s JOIN '{TRABAJADORES}' t USING (id_trabajador)
                        WHERE t.motivo_exclusion IS NOT NULL""") == 0


def test_split_en_panel_coincide(con):
    assert uno(con, f"""SELECT count(*) FROM '{PANEL}' p JOIN '{TRABAJADORES}' t USING (id_trabajador)
                        WHERE (p.split IS NULL) <> (t.motivo_exclusion IS NOT NULL)""") == 0


def test_percentil_salarial(con):
    assert uno(con, f"SELECT count(*) FROM '{PANEL}' WHERE (rem_tot = 0) <> (pct_rem_mes = 0)") == 0
    assert uno(con, f"SELECT min(pct_rem_mes) FILTER (WHERE rem_tot > 0) FROM '{PANEL}'") == 1
    assert uno(con, f"SELECT max(pct_rem_mes) FROM '{PANEL}'") == 100


def test_relaciones_cubren_el_panel(con):
    assert uno(con, f"SELECT sum(n_meses) FROM '{RELACIONES}'") == uno(con, f"SELECT count(*) FROM '{PANEL}'")
