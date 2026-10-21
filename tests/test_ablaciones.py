"""Pruebas de la fase 8: atributos sin «último valor», controles temporales y tablas de las variantes de T2-T4."""

import numpy as np
import polars as pl
import pytest

from labor2vec.tareas import derivadas
from labor2vec.tokens.datos import DatasetPretrain
from labor2vec.tokens.documento import N_FONDO, Opciones, opciones_de
from labor2vec.tokens.vocabulario import SEP

ULTIMO_VALOR = {"rama", "rama2", "provincia", "antigüedad"}


def _ds(mundo, **kw):
    return DatasetPretrain(mundo["salida"] / "eventos_train", mundo["vocab"], Opciones(max_length=2560, **kw), False)


def _doc(ds, k):
    return ds.documento(k, ds._generador(k), False)


def _eventos(doc, por_evento):
    """Bloques de tokens de cada evento (después del background) con su fecha."""
    t, d = doc.tokens[N_FONDO:], doc.abspos[N_FONDO:]
    return [(tuple(t[i: i + por_evento]), int(d[i])) for i in range(0, len(t), por_evento)]


def test_opciones_de_valores_por_defecto():
    op = opciones_de({"max_length": 128, "fin_ventana": 10})
    assert (op.atributos, op.control_temporal, op.eventos_compactos) == ("todos", None, False)
    op = opciones_de({"max_length": 128, "fin_ventana": 10, "atributos": "sin_ultimo_valor", "control_temporal": "tiempo"},
                     mask_ratio=0.2)
    assert (op.atributos, op.control_temporal, op.mask_ratio) == ("sin_ultimo_valor", "tiempo", 0.2)


def test_sin_ultimo_valor_da_tres_tokens_por_evento(mundo):
    a, c = _ds(mundo), _ds(mundo, atributos="sin_ultimo_valor")
    cat = np.array(mundo["vocab"].categorias)
    for k in range(len(a)):
        da, dc = _doc(a, k), _doc(c, k)
        n = (len(da) - N_FONDO) // 6
        assert len(dc) == N_FONDO + 3 * n and len(da) == N_FONDO + 6 * n
        assert not (set(cat[dc.tokens]) & ULTIMO_VALOR)
        assert (dc.tokens[N_FONDO + 2:: 3] == SEP).all()
        # los tokens de salario y tamaño son los mismos que en la variante A
        assert np.array_equal(dc.tokens[N_FONDO::3], da.tokens[N_FONDO::6])
        assert np.array_equal(dc.tokens[N_FONDO + 1:: 3], da.tokens[N_FONDO + 1:: 6])
        assert np.array_equal(dc.abspos[N_FONDO::3], da.abspos[N_FONDO::6])


def test_largos_coincide_con_el_documento(mundo):
    for kw in ({}, {"atributos": "sin_ultimo_valor"}, {"eventos_compactos": True}):
        ds = _ds(mundo, **kw)
        assert [len(_doc(ds, k)) for k in range(len(ds))] == ds.largos().tolist()


def test_sin_ultimo_valor_no_se_combina_con_compactos(mundo):
    ds = _ds(mundo, atributos="sin_ultimo_valor", eventos_compactos=True)
    with pytest.raises(ValueError):
        _doc(ds, 0)


def test_control_tiempo_cambia_solo_fechas_y_edades(mundo):
    base, ctrl = _ds(mundo), _ds(mundo, control_temporal="tiempo")
    movidos = 0
    for k in range(len(base)):
        a, b = _doc(base, k), _doc(ctrl, k)
        assert np.array_equal(a.tokens, b.tokens) and np.array_equal(a.segmento, b.segmento)
        assert sorted(a.abspos) == sorted(b.abspos) and sorted(a.edad) == sorted(b.edad)
        assert np.array_equal(a.abspos[:N_FONDO], b.abspos[:N_FONDO])  # el background no tiene fecha
        movidos += not np.array_equal(a.abspos, b.abspos)
        # la fecha y la edad viajan juntas: cada par (fecha, edad) del control existe en el original
        assert set(zip(b.abspos.tolist(), b.edad.tolist())) == set(zip(a.abspos.tolist(), a.edad.tolist()))
    assert movidos > 0


def test_control_orden_conserva_cada_evento_con_su_fecha(mundo):
    base, ctrl = _ds(mundo), _ds(mundo, control_temporal="orden")
    movidos = 0
    for k in range(len(base)):
        a, b = _doc(base, k), _doc(ctrl, k)
        assert sorted(_eventos(a, 6)) == sorted(_eventos(b, 6))
        movidos += _eventos(a, 6) != _eventos(b, 6)
    assert movidos > 0


@pytest.mark.parametrize("control", ["tiempo", "orden"])
def test_controles_son_deterministas_y_piden_rng(mundo, control):
    x, y = _ds(mundo, control_temporal=control), _ds(mundo, control_temporal=control)
    for k in range(len(x)):
        assert np.array_equal(_doc(x, k).como_matriz(), _doc(y, k).como_matriz())
    with pytest.raises(ValueError):
        x.documento(0, None, False)


def test_control_desconocido(mundo):
    with pytest.raises(ValueError):
        _doc(_ds(mundo, control_temporal="otro"), 0)


def test_sin_controles_el_documento_es_el_de_siempre(mundo):
    a = _ds(mundo)
    b = _ds(mundo, atributos="todos", control_temporal=None)
    assert all(np.array_equal(_doc(a, k).como_matriz(), _doc(b, k).como_matriz()) for k in range(len(a)))


# ---- tablas de las variantes de T2-T4 -------------------------------------------------------------------------------
def _tablas_base():
    ids = list(range(1, 25))
    split = ["train"] * 16 + ["val"] * 4 + ["test"] * 4
    base = {"id_trabajador": ids, "split": split, "sexo": [1] * 24, "fnacim": [1980] * 24, "edad_corte": [35] * 24}
    q15 = [i % 5 for i in range(24)]
    q19 = [min(4, max(0, q + d)) for q, d in zip(q15, [-1, 0, 1, 0, 2, 0] * 4)]
    t2 = pl.DataFrame(base | {"y": q19, "q2015": q15, "pct2015": [50.0] * 24, "pct2019": [50.0] * 24},
                      schema_overrides={"y": pl.Int8, "q2015": pl.Int8})
    # T3: la letra cambia en los ids múltiplos de 3; destinos 7, 7, 9, 11, 4, 5, 13, 14, ... (el 7 es el más frecuente)
    destinos = [7, 7, 9, 11, 4, 5, 13, 14] * 3
    letra15, letra19, y3 = [], [], []
    for i in ids:
        cambia = i % 3 == 0
        letra15.append(3)
        letra19.append(destinos[i // 3 - 1] if cambia else 3)
        y3.append(int(cambia))
    t3 = pl.DataFrame(base | {"y": y3, "letra2015": letra15, "letra2019": letra19}, schema_overrides={"y": pl.Int8})
    tr15 = [1 + i % 4 for i in range(24)]
    tr19 = [t + d for t, d in zip(tr15, [0, 1, -1, 0] * 6)]
    y4 = [0 if b > a else (2 if b < a else 1) for a, b in zip(tr15, tr19)]  # 0 más chica (tramo mayor), 2 más grande
    t4 = pl.DataFrame(base | {"y": y4, "tramo2015": tr15, "tramo2019": tr19}, schema_overrides={"y": pl.Int8})
    return t2, t3, t4


def test_variantes_de_las_tareas():
    t2, t3, t4 = _tablas_base()
    v = derivadas.variantes(t2, t3, t4)
    assert set(v) == {"t2c", "t3d", "t4c"}
    # T2c: baja / igual / sube respecto de q2015
    c = v["t2c"]
    esperado = np.sign(t2["y"].to_numpy().astype(int) - t2["q2015"].to_numpy().astype(int)) + 1
    assert c["y"].to_list() == esperado.tolist() and c.height == t2.height
    # T3d: sólo quienes cambian; el destino más frecuente en train es la clase 0 y lo raro cae en «otras» (5)
    d = v["t3d"]
    assert d.height == int(t3["y"].sum()) and set(d["y"].to_list()) <= set(range(derivadas.N_DESTINOS + 1))
    top = d.filter(pl.col("split") == "train").group_by("y").len().sort("len", descending=True)
    assert d.filter(pl.col("letra2019") == 7)["y"].unique().to_list() == [0]
    assert (d.filter(~pl.col("letra2019").is_in([7, 9, 11, 4, 5]))["y"] == derivadas.N_DESTINOS).all()
    assert top.height >= 2
    # T4c: cambia de tamaño sí/no
    f = v["t4c"]
    assert f["y"].to_list() == (t4["tramo2019"] != t4["tramo2015"]).cast(pl.Int8).to_list() and f.height == t4.height
    split = dict(zip(t2["id_trabajador"].to_list(), t2["split"].to_list()))
    for df in v.values():  # mismos splits que las tablas de origen
        assert {"id_trabajador", "split", "y"} <= set(df.columns)
        assert all(split[i] == s for i, s in zip(df["id_trabajador"].to_list(), df["split"].to_list()))


# ---- predecir con controles ------------------------------------------------------------------------------------------
def test_predecir_con_controles(mundo, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    import yaml

    from labor2vec.entrenar import cargar_config
    from labor2vec.modelo.clasificador import ClasificadorT1
    from labor2vec.modelo.config import HPARAMS_MINIMO, hparams
    from labor2vec.modelo.pretrain import TransformerEncoder
    from labor2vec.tareas import predecir

    ids = np.arange(1, 61)
    t1 = tmp_path / "t1.parquet"
    pq.write_table(pa.table({"id_trabajador": ids, "edad_corte": 30 + ids % 20, "y_2019": ids % 3 == 0,
                             "y_2021": ids % 2 == 0}), t1)
    torch.manual_seed(0)
    hp = hparams(HPARAMS_MINIMO, vocab_size=len(mundo["vocab"]), max_length=128, n_tokens_fondo=4)
    modelo = ClasificadorT1(dict(TransformerEncoder(hp).hparams)).eval()
    with torch.no_grad():  # las compuertas ReZero de edad y fecha arrancan en 0: sin esto el modelo no mira el tiempo
        modelo.encoder.embedding.res_age.weights.fill_(1.0)
        modelo.encoder.embedding.res_abs.weights.fill_(1.0)
    corrida = tmp_path / "t1_l2v_s0"
    corrida.mkdir()
    torch.save({"state_dict": modelo.state_dict(), "hparams": dict(modelo.hparams)}, corrida / "modelo_final.pt")
    cfg = cargar_config("configs/ajuste_t1.yaml", [
        f"datos.corpus={mundo['salida']}", f"datos.t1={t1}", "datos.tamano_lote=4", "datos.max_length=128",
        "datos.num_workers=0", "datos.fin_ventana=39"])
    ruta = tmp_path / "cfg.yaml"
    ruta.write_text(yaml.safe_dump(cfg))

    predecir.main(["--config", str(ruta), "--corridas", str(corrida), "--runs", str(tmp_path)])
    p = {}
    for c in predecir.CONTROLES:
        t = pq.read_table(tmp_path / f"ctrl_{c}" / "t1_l2v_s0" / "predicciones_test.parquet").to_pydict()
        assert set(t) == {"id_trabajador", "y_2019", "y_2021", "p"} and len(t["p"]) == 10
        p[c] = dict(zip(t["id_trabajador"], t["p"]))
    assert p["ninguno"].keys() == p["tiempo"].keys() == p["orden"].keys()
    # sin control las predicciones son las del modelo sobre la entrada de siempre
    dm = predecir.datamodule(cfg, "ninguno")
    directas = modelo.predecir(dm.test_dataloader())
    assert np.allclose([p["ninguno"][int(i)] for i in directas["id"]], directas["p"], atol=1e-6)
    # los controles cambian la entrada, así que cambian las predicciones
    assert any(abs(p["tiempo"][i] - p["ninguno"][i]) > 1e-6 for i in p["ninguno"])
    assert any(abs(p["orden"][i] - p["ninguno"][i]) > 1e-6 for i in p["ninguno"])


# ---- comparaciones de las ablaciones ---------------------------------------------------------------------------------
def _escribir_preds(carpeta, ids, y, p):
    import pyarrow as pa
    import pyarrow.parquet as pq

    carpeta.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"id_trabajador": ids, "y": y, "p": p}), carpeta / "predicciones_test.parquet")


def test_bootstrap_diferencia():
    from labor2vec.tareas import metricas

    rng = np.random.default_rng(0)
    y = (rng.random(3000) < 0.3).astype(int)
    bueno = np.clip(0.3 * y + rng.random(3000) * 0.7, 0, 1)
    malo = rng.random(3000)
    d, lo, hi = metricas.bootstrap_diferencia(y, [bueno], [malo], metricas.auc, n=200)
    assert lo < d < hi and lo > 0.1
    d0, lo0, hi0 = metricas.bootstrap_diferencia(y, [bueno], [bueno], metricas.auc, n=50)
    assert d0 == lo0 == hi0 == 0.0


def test_comparaciones_sin_ultimo_valor_y_continuidad(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from labor2vec.tareas import ablaciones

    rng = np.random.default_rng(1)
    n = 2000
    ids = np.arange(1, n + 1)
    y = (rng.random(n) < 0.2).astype(int)
    continua = rng.random(n) < 0.5
    ruido = rng.random(n)
    con = np.clip(0.5 * y + 0.5 * ruido, 0, 1)
    # sin los atributos el modelo pierde señal sólo entre quienes continúan (donde «el futuro se colaba»)
    sin = np.where(continua, ruido, con)
    _escribir_preds(tmp_path / "t3_l2v_s0", ids, y, con)
    _escribir_preds(tmp_path / "t3_uv_l2v_s0", ids, y, sin)
    ruta = tmp_path / "continuidad.parquet"
    pq.write_table(pa.table({"id_trabajador": ids, "continua": continua}), ruta)

    res: dict = {}
    texto = ablaciones.sin_ultimo_valor(tmp_path, 100, res)
    r = res["sin_ultimo_valor"]["t3_Transformer pre-entrenado"]
    assert r["dif"] > 0.05 and r["lo"] > 0 and "T3: con y sin atributos de último valor" in texto
    texto = ablaciones.continuidad(tmp_path, 100, res, ruta)
    c, t = res["continuidad"]["t3_Transformer pre-entrenado_continua"], res["continuidad"]["t3_Transformer pre-entrenado_termino"]
    assert c["dif"] > 0.1 and c["lo"] > 0          # la diferencia está donde continúa la relación
    assert abs(t["dif"]) < 1e-9                    # y no donde terminó antes
    assert "continúa después del corte" in texto


def test_comparaciones_de_control_y_reajuste(tmp_path):
    from labor2vec.tareas import ablaciones

    rng = np.random.default_rng(2)
    n = 1500
    ids = np.arange(1, n + 1)
    y = (rng.random(n) < 0.2).astype(int)
    bueno = np.clip(0.5 * y + 0.5 * rng.random(n), 0, 1)
    for carpeta, p in (("ctrl_ninguno/t3_l2v_s0", bueno), ("ctrl_tiempo/t3_l2v_s0", rng.random(n)),
                       ("t3_l2v_s0", bueno), ("t3_tiempo_l2v_s0", bueno)):
        _escribir_preds(tmp_path / carpeta, ids, y, p)
    res: dict = {}
    assert "Control «tiempo»" in ablaciones.control_temporal(tmp_path, 50, res)
    assert res["control_temporal"]["t3_tiempo"]["dif"] > 0.1
    assert "orden" not in "".join(res["control_temporal"])  # sin corridas de ese control, no hay fila
    ablaciones.reajuste_tiempo(tmp_path, 50, res)
    assert res["reajuste_tiempo"]["t3"]["dif"] == 0.0


def test_sql_continuidad(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from labor2vec.tareas import derivadas
    from labor2vec.utiles.duck import conectar

    t1 = tmp_path / "t1.parquet"
    pq.write_table(pa.table({"id_trabajador": [1, 2, 3]}), t1)
    filas = []
    for mes in range(228, 240):               # 1: una sola relación que sigue hasta 2017
        filas.append((1, 1, 100.0, mes))
    for mes in range(240, 260):
        filas.append((1, 1, 100.0, mes))
    for mes in range(228, 240):               # 2: la principal termina en 2015-12; otra empieza después pero es corta
        filas.append((2, 1, 100.0, mes))
    for mes in range(238, 245):
        filas.append((2, 2, 50.0, mes))
    for mes in range(228, 234):               # 3: relación larga que terminó a mitad de 2015 y otra que sigue
        filas.append((3, 1, 100.0, mes))
    for mes in range(232, 250):
        filas.append((3, 2, 120.0, mes))
    panel = tmp_path / "panel.parquet"
    pl.DataFrame(filas, schema=["id_trabajador", "id_relacion", "rem_tot", "mes_idx"], orient="row").write_parquet(panel)
    df = conectar().execute(derivadas.sql_continuidad(panel, t1)).pl().sort("id_trabajador")
    # 1: continúa; 2: la relación 1 (12 meses) es la principal y termina en 2015-12; 3: la 2 (8 meses en 2015) vence a la 1 (6)
    assert df["continua"].to_list() == [True, False, True]
    assert df["ultimo_mes"].to_list() == [259, 239, 249]


# ---- de punta a punta: pre-entrenar y ajustar con las opciones nuevas -------------------------------------------------
def test_pretrain_y_ajuste_sin_ultimo_valor_con_control(mundo, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    import yaml

    from labor2vec import ajustar, entrenar
    from labor2vec.entrenar import cargar_config

    cfg = cargar_config("configs/humo_cpu.yaml", [
        f"datos.corpus={mundo['salida']}", "datos.tamano_lote=8", "datos.max_length=128", "datos.atributos=sin_ultimo_valor",
        "entrenamiento.epocas=1", "entrenamiento.pasos_por_epoca=2", "entrenamiento.val_lotes=2",
        "entrenamiento.guardar_cada_pasos=2", "entrenamiento.muestra_unigrama=20",
        f"entrenamiento.salida={tmp_path / 'pre'}"])
    ruta = tmp_path / "pre.yaml"
    ruta.write_text(yaml.safe_dump(cfg))
    entrenar.main(["--config", str(ruta)])
    pre = tmp_path / "pre" / "modelo_final.pt"
    assert pre.exists()
    dm, _ = entrenar.preparar(cfg)  # el dataset de pre-entrenamiento ve eventos de 3 tokens
    cat = np.array(mundo["vocab"].categorias)
    ds = dm.ds["train"]
    for k in range(5):  # documento sin enmascarar (el MLM puede meter tokens al azar de cualquier categoría)
        doc = ds.documento(k, ds._generador(k), False)
        assert not (set(cat[doc.tokens]) & ULTIMO_VALOR) and len(doc) == N_FONDO + 3 * ((len(doc) - N_FONDO) // 3)

    ids = np.arange(1, 61)
    t1 = tmp_path / "t1.parquet"
    pq.write_table(pa.table({"id_trabajador": ids, "edad_corte": 30 + ids % 20, "y_2019": ids % 3 == 0,
                             "y_2021": ids % 2 == 0}), t1)
    cfg = cargar_config("configs/ajuste_t1.yaml", [
        f"datos.corpus={mundo['salida']}", f"datos.t1={t1}", "datos.tamano_lote=4", "datos.max_length=128",
        "datos.num_workers=0", "datos.fin_ventana=10", "datos.atributos=sin_ultimo_valor", "datos.control_temporal=tiempo",
        f"preentrenado={pre}", "entrenamiento.precision=32-true", "entrenamiento.acumular=1",
        "entrenamiento.pasos_por_epoca=3", "entrenamiento.epocas_min=1", "entrenamiento.epocas_max=2",
        "entrenamiento.guardar_cada_pasos=2", "ajuste.lr=0.001", f"entrenamiento.salida={tmp_path / 'aj'}"])
    ruta = tmp_path / "aj.yaml"
    ruta.write_text(yaml.safe_dump(cfg))
    ajustar.main(["--config", str(ruta)])
    assert (tmp_path / "aj" / "modelo_final.pt").exists()
    t = pq.read_table(tmp_path / "aj" / "predicciones_test.parquet").to_pydict()
    assert len(t["p"]) == 10 and all(0.0 <= p <= 1.0 for p in t["p"])
    assert torch.load(tmp_path / "aj" / "modelo_final.pt", weights_only=False)["hparams"]["hp_encoder"]["vocab_size"] == len(mundo["vocab"])
