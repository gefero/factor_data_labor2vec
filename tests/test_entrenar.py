"""Pruebas de la fase 3: métricas, línea de base unigrama y script de entrenamiento."""

import numpy as np
import pytest
import torch

from labor2vec.entrenar import cargar_config, main
from labor2vec.modelo.config import HPARAMS_MINIMO, hparams
from labor2vec.modelo.pretrain import TransformerEncoder
from labor2vec.tokens.datos import DataModuleMLER, DatasetPretrain
from labor2vec.tokens.documento import Opciones
from labor2vec.tokens.linea_base import logp_unigrama
from labor2vec.tokens.vocabulario import N_ESPECIALES


def test_unigrama_suma_uno_y_excluye_especiales(mundo):
    v = mundo["vocab"]
    ds = DatasetPretrain(mundo["salida"] / "eventos_train", v, Opciones(), False)
    lp = logp_unigrama(ds, len(v), n_docs=20)
    assert lp.shape == (len(v),)
    assert torch.exp(lp).sum() == pytest.approx(1.0, abs=1e-3)
    assert (torch.exp(lp[:N_ESPECIALES]) < 1e-9).all()
    # el token más frecuente (separador de eventos aparte) tiene más masa que uno que no aparece
    assert lp.max() > lp[N_ESPECIALES:].min()


def _modelo_chico(v, logp=None):
    hp = hparams(HPARAMS_MINIMO, vocab_size=len(v), max_length=128, n_tokens_fondo=4,
                 sop_proporciones=(0.9, 0.05, 0.05))
    torch.manual_seed(0)
    m = TransformerEncoder(hp)
    if logp is not None:
        m.fijar_unigrama(logp)
    return m


def test_metricas_a_mano(mundo):
    """Con logits conocidos: perplejidad = exp(nll medio), exactitud y recall por clase; el relleno no cuenta."""
    v = mundo["vocab"]
    m = _modelo_chico(v, torch.log(torch.full((len(v),), 1.0 / len(v))))
    V = len(v)
    logits = torch.full((2, 3, V), -10.0)
    objetivos = torch.tensor([[12, 13, 0], [14, 0, 0]])  # los ceros son relleno
    for (i, j), t in {(0, 0): 12, (0, 1): 20, (1, 0): 14}.items():
        logits[i, j, t] = 10.0  # acierta 2 de 3
    batch = {"target_tokens": objetivos, "target_sop": torch.tensor([0, 1])}
    sop = torch.tensor([[5.0, 0, 0], [5.0, 0, 0]])  # predice siempre 0
    r = m._metricas(logits, sop, batch)
    assert r["n"] == 3 and r["aciertos"] == 2 and r["sop_ok"] == 1
    esperado = torch.nn.functional.cross_entropy(logits[[0, 0, 1], [0, 1, 0]], torch.tensor([12, 13, 14]),
                                                 reduction="sum")
    assert r["nll"] == pytest.approx(esperado.item(), rel=1e-5)
    assert r["nll_unigrama"] == pytest.approx(3 * np.log(V), rel=1e-5)


def test_validacion_registra_las_metricas(mundo):
    import lightning.pytorch as pl

    v = mundo["vocab"]
    dm = DataModuleMLER(mundo["salida"], v, Opciones(max_length=128), tamano_lote=8, num_workers=0)
    dm.setup()
    m = _modelo_chico(v, logp_unigrama(dm.ds["train"], len(v), 20))
    t = pl.Trainer(max_epochs=1, limit_train_batches=3, limit_val_batches=3, logger=False,
                   enable_checkpointing=False, enable_progress_bar=False, accelerator="cpu")
    t.fit(m, datamodule=dm)
    for k in ("val/mlm_loss", "val/mlm_ppl", "val/ppl_unigrama", "val/mlm_acc", "val/sop_acc", "val/sop_recall_c0",
              "train/loss"):
        assert k in t.callback_metrics and torch.isfinite(t.callback_metrics[k])


def test_lotes_reproducibles_con_workers(mundo):
    """Misma semilla, mismos lotes de validación, con y sin procesos de carga."""
    def lotes(workers):
        dm = DataModuleMLER(mundo["salida"], mundo["vocab"], Opciones(max_length=128), 8, workers, semilla=5)
        dm.setup()
        return [b["input_ids"] for _, b in zip(range(3), dm.val_dataloader())]

    a, b = lotes(0), lotes(2)
    assert all(torch.equal(x, y) for x, y in zip(a, b))


# ---- script de entrenamiento -----------------------------------------------------------------------------
def _config(mundo, tmp_path):
    cfg = cargar_config("configs/humo_cpu.yaml", [
        f"datos.corpus={mundo['salida']}", "datos.tamano_lote=8", "datos.max_length=128", "entrenamiento.epocas=2",
        "entrenamiento.pasos_por_epoca=3", "entrenamiento.val_lotes=2", "entrenamiento.guardar_cada_pasos=2",
        "entrenamiento.muestra_unigrama=20", f"entrenamiento.salida={tmp_path / 'run'}"])
    ruta = tmp_path / "cfg.yaml"
    import yaml
    ruta.write_text(yaml.safe_dump(cfg))
    return ruta


def test_entrenar_escribe_checkpoint_y_reanuda(mundo, tmp_path):
    ruta = _config(mundo, tmp_path)
    run = tmp_path / "run"
    main(["--config", str(ruta), "--max-pasos", "4"])  # corta a mitad de la segunda época
    ck = torch.load(run / "last.ckpt", map_location="cpu", weights_only=False)
    assert ck["global_step"] == 4 and ck["epoch"] == 1
    assert (run / "mejor.ckpt").exists() and (run / "config.json").exists()
    main(["--config", str(ruta), "--reanudar"])  # mismo comando: sigue hasta completar las 2 épocas (6 pasos)
    ck2 = torch.load(run / "last.ckpt", map_location="cpu", weights_only=False)
    assert ck2["global_step"] == 6 and ck2["epoch"] == 2


def test_entrenar_se_detiene_por_tiempo(mundo, tmp_path):
    ruta = _config(mundo, tmp_path)
    main(["--config", str(ruta), "--set", "entrenamiento.epocas=1000", "--max-minutos", "0.02"])
    ck = torch.load(tmp_path / "run" / "last.ckpt", map_location="cpu", weights_only=False)
    assert 0 < ck["global_step"] < 3000


def test_el_camino_de_entrenamiento_no_importa_polars():
    """El binario de polars exige AVX2 y los nodos de Mendieta no lo tienen (Illegal instruction)."""
    import subprocess
    import sys

    codigo = ("import sys, labor2vec.entrenar, labor2vec.tokens.datos, labor2vec.tokens.vocabulario, "
              "labor2vec.tokens.corpus; sys.exit(1 if 'polars' in sys.modules else 0)")
    assert subprocess.run([sys.executable, "-I", "-c", codigo]).returncode == 0
