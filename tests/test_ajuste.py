"""Pruebas de la fase 6: dataset de T1, clasificador y script de ajuste."""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import yaml

from labor2vec import ajustar
from labor2vec.entrenar import cargar_config
from labor2vec.modelo.clasificador import AttentionDecoder, ClasificadorT1, perdida_balanceada
from labor2vec.modelo.config import HPARAMS_MINIMO, hparams
from labor2vec.modelo.pretrain import TransformerEncoder
from labor2vec.tareas.datos import DataModuleT1, DatasetT1, SamplerBalanceado, collate
from labor2vec.tokens.documento import Opciones, dias_de_mes

CORTE = 10  # fin de la ventana de las pruebas (el panel sintético llega hasta el mes ~40)


@pytest.fixture(scope="module")
def t1(mundo, tmp_path_factory):
    """Cohorte sintética: los 60 trabajadores del panel, con etiqueta y_2019 = (id % 3 == 0)."""
    ids = np.arange(1, 61)
    ruta = tmp_path_factory.mktemp("t1") / "t1.parquet"
    pq.write_table(pa.table({"id_trabajador": ids, "edad_corte": 30 + ids % 20, "y_2019": ids % 3 == 0,
                             "y_2021": ids % 2 == 0}), ruta)
    return ruta


@pytest.fixture(scope="module")
def pre(mundo, tmp_path_factory):
    """Un 'modelo_final.pt' chico, como el que escribe `entrenar`."""
    hp = hparams(HPARAMS_MINIMO, vocab_size=len(mundo["vocab"]), max_length=128, n_tokens_fondo=4)
    torch.manual_seed(0)
    m = TransformerEncoder(hp)
    ruta = tmp_path_factory.mktemp("pre") / "modelo_final.pt"
    torch.save({"state_dict": m.state_dict(), "hparams": dict(m.hparams)}, ruta)
    return ruta


def _dataset(mundo, t1, split="train", augmentar=False):
    return DatasetT1(mundo["salida"] / f"eventos_{split}", mundo["vocab"],
                     Opciones(max_length=128, fin_ventana=CORTE), augmentar, t1)


def test_dataset_solo_cohorte_y_etiquetas_alineadas(mundo, t1, tmp_path):
    ds = _dataset(mundo, t1)
    assert set(ds.id) <= set(range(1, 41)) and len(ds) == len(ds.id) > 0
    assert (ds.y == (ds.id % 3 == 0)).all()
    muestra = ds[0]
    assert muestra["id"] == ds.id[0] and muestra["target"] == ds.y[0]
    assert len(ds.largos()) == len(ds)
    # una cohorte más chica deja afuera a los demás
    ids = np.arange(1, 21)
    chica = tmp_path / "chica.parquet"
    pq.write_table(pa.table({"id_trabajador": ids, "edad_corte": 30 + ids % 20, "y_2019": ids % 3 == 0,
                             "y_2021": ids % 2 == 0}), chica)
    assert set(_dataset(mundo, chica).id) <= set(ids) and _dataset(mundo, chica, "val").id.size == 0


def test_el_documento_no_tiene_eventos_posteriores_al_corte(mundo, t1):
    ds = _dataset(mundo, t1)
    for k in range(len(ds)):
        m = ds[k]["input_ids"]
        assert m[1, 4:].max() <= dias_de_mes(np.array([CORTE]))[0]  # abspos de los eventos
        assert (m[0] != 0).all() and m.shape[0] == 4


def test_documentos_deterministas_sin_augmentar(mundo, t1):
    a, b = _dataset(mundo, t1), _dataset(mundo, t1)
    assert all(np.array_equal(a[k]["input_ids"], b[k]["input_ids"]) for k in range(len(a)))


def test_collate_rellena_y_marca(mundo, t1):
    ds = _dataset(mundo, t1)
    lote = collate([ds[0], ds[1], ds[2]], max_length=128, multiplo=16)
    assert lote["input_ids"].shape[:2] == (3, 4) and lote["input_ids"].shape[2] % 16 == 0
    n = [ds[i]["input_ids"].shape[1] for i in range(3)]
    assert lote["padding_mask"].sum(1).tolist() == n
    assert lote["target"].tolist() == ds.y[:3].tolist() and lote["id"].tolist() == ds.id[:3].tolist()


def test_sampler_balanceado_da_lotes_mitad_y_mitad():
    rng = np.random.default_rng(0)
    y = (rng.random(4000) < 0.1).astype(int)
    s = SamplerBalanceado(rng.integers(10, 200, 4000), y, 32, semilla=1)
    lotes = list(s)
    assert len(lotes) == len(s) == 4000 // 32 and all(len(b) == 32 for b in lotes)
    frac = y[np.concatenate(lotes)].mean()
    assert 0.46 < frac < 0.54
    assert lotes != list(s)  # otra época, otros lotes


def test_datamodule_loaders(mundo, t1):
    dm = DataModuleT1(mundo["salida"], mundo["vocab"], Opciones(max_length=128, fin_ventana=CORTE), t1,
                      tamano_lote=4, num_workers=0)
    dm.setup()
    b = next(iter(dm.train_dataloader()))
    assert set(b) == {"input_ids", "padding_mask", "target", "id"}
    ids_test = np.concatenate([x["id"].numpy() for x in dm.test_dataloader()])  # val/test completos
    assert sorted(ids_test) == sorted(dm.ds["test"].id)


def test_cabeza_ignora_el_relleno():
    torch.manual_seed(0)
    cab = AttentionDecoder(16).eval()
    x = torch.randn(2, 6, 16)
    mascara = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    a = cab(x, mascara)
    x2 = x.clone()
    x2[0, 4:] = 100.0
    assert torch.allclose(a[0], cab(x2, mascara)[0], atol=1e-5)
    assert a.shape == (2, 2)


def test_perdida_balanceada_a_mano():
    logits = torch.tensor([[0.0, 2.0], [1.0, 0.0], [0.5, 0.0], [0.0, 0.0]])
    y = torch.tensor([1, 0, 0, 1])
    nll = torch.nn.functional.cross_entropy(logits, y, reduction="none")
    esperado = 0.5 * nll[[0, 3]].mean() + 0.5 * nll[[1, 2]].mean()
    assert perdida_balanceada(logits, y).item() == pytest.approx(esperado.item())
    assert perdida_balanceada(logits[1:3], y[1:3]).item() == pytest.approx(nll[1:3].mean().item())  # sin positivos


def test_cargar_preentrenado_copia_el_encoder(pre):
    ck = torch.load(pre, weights_only=False)
    m = ClasificadorT1(ck["hparams"])
    m.cargar_preentrenado(pre)
    pesos = {k[len("transformer."):]: v for k, v in ck["state_dict"].items() if k.startswith("transformer.")}
    for k, v in m.encoder.state_dict().items():
        assert torch.equal(v, pesos[k])


def test_solo_entrenan_cls_sep_y_unk_entre_los_embeddings(mundo, t1, pre):
    ck = torch.load(pre, weights_only=False)
    m = ClasificadorT1(ck["hparams"])
    ds = _dataset(mundo, t1)
    lote = collate([ds[i] for i in range(4)], 128, 16)
    perdida_balanceada(m(lote), lote["target"]).backward()
    original = m.encoder.embedding.token.parametrizations.weight.original
    filas = (original.grad.abs().sum(1) > 0).nonzero().flatten().tolist()
    assert set(filas) <= {1, 2, 9} and len(filas) > 0
    libre = ClasificadorT1(ck["hparams"], {"congelar_embeddings": False})
    perdida_balanceada(libre(lote), lote["target"]).backward()
    g = libre.encoder.embedding.token.parametrizations.weight.original.grad
    assert (g.abs().sum(1) > 0).sum() > 3


def _cfg(mundo, t1, pre, tmp_path, cambios=()):
    cfg = cargar_config("configs/ajuste_t1.yaml", [
        f"datos.corpus={mundo['salida']}", f"datos.t1={t1}", "datos.tamano_lote=4", "datos.max_length=128",
        "datos.num_workers=0", f"datos.fin_ventana={CORTE}", f"preentrenado={pre}", "entrenamiento.precision=32-true",
        "entrenamiento.acumular=1", "entrenamiento.pasos_por_epoca=3", "entrenamiento.epocas_min=1",
        "entrenamiento.epocas_max=2", "entrenamiento.guardar_cada_pasos=2", "ajuste.lr=0.001",
        f"entrenamiento.salida={tmp_path / 'run'}", *cambios])
    ruta = tmp_path / "cfg.yaml"
    ruta.write_text(yaml.safe_dump(cfg))
    return ruta


@pytest.mark.parametrize("cambios", [(), ("ajuste.desde_cero=true",), ("ajuste.tipo=gru",)])
def test_ajustar_de_punta_a_punta(mundo, t1, pre, tmp_path, cambios):
    """Cada variante entrena unos pasos, escribe el modelo y las predicciones, y el modelo se puede recargar."""
    ruta = _cfg(mundo, t1, pre, tmp_path, cambios)
    ajustar.main(["--config", str(ruta)])
    run = tmp_path / "run"
    assert (run / "modelo_final.pt").exists() and (run / "last.ckpt").exists()
    for split, n in (("val", 10), ("test", 10)):
        t = pq.read_table(run / f"predicciones_{split}.parquet").to_pydict()
        assert len(t["p"]) == n and all(0.0 <= p <= 1.0 for p in t["p"])
        assert set(t) == {"id_trabajador", "y_2019", "y_2021", "p"}
    ck = torch.load(run / "modelo_final.pt", weights_only=False)
    m = ClasificadorT1(ck["hparams"]["hp_encoder"], ck["hparams"]["ft"])
    m.load_state_dict(ck["state_dict"])


# ---- líneas de base tabulares y camino sin polars ------------------------------------------------------------------
def test_atributos_son_proporciones(mundo, t1):
    from labor2vec.tareas.tabulares import atributos
    ds = _dataset(mundo, t1)
    x = atributos(ds, len(mundo["vocab"]))
    assert x.shape == (len(ds), len(mundo["vocab"]) - 10 + 2)
    assert np.allclose(x[:, :-2].sum(1), 1.0, atol=1e-5) and (x[:, :-2] >= 0).all()
    assert set(np.unique(x[:, -2])) <= {0.0, 1.0} and (x[:, -1] >= 0).all()


def test_tabulares_de_punta_a_punta(mundo, t1, tmp_path):
    from labor2vec.tareas import tabulares
    datos = tabulares.cargar(mundo["salida"], t1, fin_ventana=CORTE, max_length=128)
    for modelo in (lambda d: tabulares.logistica(d, columnas=[-2, -1]), tabulares.logistica,
                   lambda d: tabulares.ffnn(d, 0, epocas=2)):
        p_val, p_test = modelo(datos)
        assert len(p_val) == len(datos["val"]["y"]) and ((0 <= p_test) & (p_test <= 1)).all()
    tabulares.escribir(tmp_path / "m", datos, p_val, p_test, t1)
    t = pq.read_table(tmp_path / "m" / "predicciones_test.parquet").to_pydict()
    assert len(t["p"]) == len(datos["test"]["y"]) and set(t) == {"id_trabajador", "y_2019", "y_2021", "p"}


def test_ajuste_baja_la_perdida(mundo, t1, pre):
    """Sobreajuste de un lote: con unos pasos de optimización la pérdida baja. Se descongelan los embeddings porque el
    modelo chico de la prueba no está pre-entrenado y los suyos son diminutos (en el real tienen desvío ~0,24)."""
    ck = torch.load(pre, weights_only=False)
    torch.manual_seed(0)
    m = ClasificadorT1(ck["hparams"], {"lr": 0.01, "congelar_embeddings": False})
    ds = _dataset(mundo, t1)
    lote = collate([ds[i] for i in range(8)], 128, 16)
    lote["target"] = (torch.arange(8) % 2).long()
    opt = m.configure_optimizers()["optimizer"]
    m.train()
    perdidas = []
    for _ in range(60):
        opt.zero_grad()
        p = perdida_balanceada(m(lote), lote["target"])
        p.backward()
        opt.step()
        perdidas.append(p.item())
    assert perdidas[-1] < 0.8 * perdidas[0]


def test_el_camino_de_ajuste_no_importa_polars():
    import subprocess
    import sys

    codigo = ("import sys, labor2vec.ajustar, labor2vec.tareas.datos, labor2vec.tareas.metricas; "
              "sys.exit(1 if 'polars' in sys.modules else 0)")
    assert subprocess.run([sys.executable, "-I", "-c", codigo]).returncode == 0


def test_evaluar_arma_tablas_y_figuras(tmp_path, monkeypatch):
    from labor2vec.tareas import evaluar

    rng = np.random.default_rng(0)
    n = 600
    ids = np.arange(n)
    y19, y21 = rng.random(n) < 0.2, rng.random(n) < 0.3
    t1_ruta = tmp_path / "t1.parquet"
    pq.write_table(pa.table({"id_trabajador": ids, "sexo": 1 + ids % 2, "edad_corte": 25 + ids % 31}), t1_ruta)
    runs = tmp_path / "runs"
    for carpeta, ruido in (("t1_l2v_s0", 0.5), ("t1_l2v_s1", 0.6), ("t1_tab/logistica", 1.5)):
        (runs / carpeta).mkdir(parents=True)
        p = 1 / (1 + np.exp(-(y19 + rng.normal(size=n) * ruido - 0.5)))
        perm = rng.permutation(ids)  # el orden de filas no importa: se alinea por id, y las etiquetas viajan con él
        pq.write_table(pa.table({"id_trabajador": perm, "y_2019": y19[perm], "y_2021": y21[perm], "p": p[perm]}),
                       runs / carpeta / "predicciones_test.parquet")
    modelos = evaluar.cargar_modelos(runs)
    assert set(modelos) == {"Transformer pre-entrenado (L2V)", "Regresión logística"}
    assert [len(v) for v in modelos.values()] == [2, 1]
    texto, res = evaluar.tabla(modelos, "y_2019", 30, {})
    assert "| Azar |" in texto and "| Mayoría |" in texto and res["Mayoría"]["c_mcc"][0] == 0.0
    assert abs(res["Azar"]["auc"][0] - 0.5) < 0.1
    assert "sd_aul" in res["Transformer pre-entrenado (L2V)"]
    assert "mujeres" in evaluar.subgrupos(modelos, t1_ruta) and "Brier" in evaluar.calibracion(modelos)
    evaluar.figuras(modelos, tmp_path)
    assert (tmp_path / "curvas.png").stat().st_size > 1000
