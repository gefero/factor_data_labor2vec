"""Pruebas de la fase 7: sonda (vectores, gradientes, SmoothGrad), TCAV y tabla de conceptos."""

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from labor2vec.analisis import extraer, personas, tcav
from labor2vec.modelo.clasificador import ClasificadorT1
from labor2vec.modelo.config import HPARAMS_MINIMO, hparams
from labor2vec.modelo.pretrain import TransformerEncoder
from labor2vec.tareas.datos import DatasetT1, collate
from labor2vec.tokens.documento import Opciones

CORTE = 10


@pytest.fixture(scope="module")
def modelo(mundo):
    hp = hparams(HPARAMS_MINIMO, vocab_size=len(mundo["vocab"]), max_length=128, n_tokens_fondo=4)
    torch.manual_seed(0)
    enc = TransformerEncoder(hp)
    m = ClasificadorT1(dict(enc.hparams))
    m.encoder.load_state_dict({k[len("transformer."):]: v for k, v in enc.state_dict().items() if k.startswith("transformer.")})
    return m.eval()


@pytest.fixture(scope="module")
def lote(mundo, tmp_path_factory):
    ids = np.arange(1, 61)
    ruta = tmp_path_factory.mktemp("t1") / "t1.parquet"
    pq.write_table(pa.table({"id_trabajador": ids, "edad_corte": 30 + ids % 20, "y_2019": ids % 3 == 0,
                             "y_2021": ids % 2 == 0}), ruta)
    ds = DatasetT1(mundo["salida"] / "eventos_train", mundo["vocab"], Opciones(max_length=128, fin_ventana=CORTE), False, ruta)
    return ds, collate([ds[i] for i in range(6)], 128, 16)


def test_la_sonda_reproduce_el_logit_del_modelo(modelo, lote):
    _, b = lote
    sonda = extraer.Sonda(modelo, sigma=0.5)
    sal = sonda.pasar(b)
    lg = modelo(b)
    assert torch.allclose(sal["logit"], lg[:, 1] - lg[:, 0], atol=1e-5)  # el ruido de SmoothGrad no se cuela en pasar()
    assert sal["vec"].shape == (6, modelo.encoder.embedding.token.weight.shape[1])


def test_g_es_la_derivada_al_sumar_un_vector_a_todos_los_tokens(modelo, lote):
    _, b = lote
    sonda = extraer.Sonda(modelo)
    g = sonda.pasar(b, con_grad=True)["g_ult"]
    v = torch.randn(g.shape[1])
    v = v / v.norm()
    mascara = b["padding_mask"].long()
    eps = 1e-2

    def logit(delta):
        h = modelo.encoder(b["input_ids"].long(), mascara) + delta * v * mascara.unsqueeze(-1)
        lg = modelo.cabeza(h, mascara)
        return lg[:, 1] - lg[:, 0]

    with torch.no_grad():
        numerica = (logit(eps) - logit(-eps)) / (2 * eps)
    assert torch.allclose(numerica, g @ v, rtol=1e-2, atol=1e-3)


def test_smoothgrad_puntaje_por_token_real_suma_uno(modelo, lote):
    _, b = lote
    sonda = extraer.Sonda(modelo, sigma=extraer.sigma_ruido(modelo, 0.15))
    s = sonda.smoothgrad(b, n_copias=3)
    assert s.shape == b["padding_mask"].shape and (s >= 0).all()
    assert torch.allclose(s.sum(1), torch.ones(len(s)), atol=1e-5)
    assert (s[~b["padding_mask"].bool()] == 0).all()


def test_muestra_estratificada_cubre_las_celdas():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.2).astype(int)
    logit = rng.normal(size=5000) + y
    i = extraer.muestra_estratificada(y, logit, 400)
    assert 300 < len(i) <= 400 and 0.3 < y[i].mean() < 0.7  # sobre-representa los positivos


def test_tcav_distingue_un_concepto_que_importa_de_uno_que_no():
    rng = np.random.default_rng(0)
    n, d = 3000, 10
    x = rng.normal(size=(n, d))
    importa = rng.random(n) < 0.3
    inerte = rng.random(n) < 0.3
    x[importa, 0] += 3.0  # el concepto «importa» es la dirección del eje 0
    x[inerte, 1] += 3.0   # el «inerte» es la del eje 1
    y = (rng.random(n) < 0.5).astype(int)
    g = rng.normal(size=(n, d)) * 0.1
    g[:, 0] += 1.0  # el logit sube con el eje 0 en todas las personas; el eje 1 no cuenta
    filas = tcav.tabla(x, g, y, {"importa": importa, "inerte": inerte}, n=20, tam=300)
    por = {(f["concepto"], f["clase"]): f for f in filas}
    assert por[("importa", 1)]["tcav"] > 0.95 and por[("importa", 1)]["p"] < 1e-3
    assert por[("inerte", 1)]["p"] > 0.01  # indistinguible de las direcciones al azar
    assert por[("importa", 1)]["exactitud"] > 0.9


def test_tabla_de_conceptos_una_fila_por_persona(tmp_path):
    """Panel mínimo que llega hasta el corte real (mes 239): relación principal = la de más meses en 2015."""
    ids = np.arange(1, 41)
    t1 = tmp_path / "t1.parquet"
    pq.write_table(pa.table({"id_trabajador": ids, "split": ["train"] * 40, "sexo": 1 + ids % 2,
                             "edad_corte": 30 + ids % 20}), t1)
    filas = []
    for t in ids:
        for mes in range(220, 240):
            filas.append((t, 1, 3 + t % 3, 1 + t % 4, 100.0, 10 + t, mes))  # relación larga
            if t == 1 and mes >= 236:
                filas.append((t, 2, 9, 1, 50.0, 90, mes))  # relación corta: no es la principal
    panel = tmp_path / "panel.parquet"
    pl.DataFrame(filas, schema=["id_trabajador", "id_relacion", "letra", "tramo_empleo", "rem_tot", "pct_rem_mes", "mes_idx"],
                 orient="row").write_parquet(panel)
    from labor2vec.utiles.duck import conectar
    df = conectar().execute(personas.sql_conceptos(panel, t1)).pl()
    assert df.height == 40 and df["id_trabajador"].n_unique() == 40
    fila1 = df.filter(pl.col("id_trabajador") == 1).row(0, named=True)
    assert fila1["letra"] == 4 and fila1["antig_anios"] == pytest.approx(19 / 12) and fila1["pct2015"] > 0
    etiquetas, grupos = personas.conceptos_de(df)
    assert all(len(v) == 40 for v in etiquetas.values()) and all(len(v) == 40 for v in grupos.values())
    assert grupos["mujer"].sum() == 20
