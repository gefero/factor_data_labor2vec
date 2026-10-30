"""Prueba de humo del backbone de life2vec en CPU."""

import torch

from labor2vec.modelo.config import HPARAMS_MINIMO, hparams, lote_aleatorio
from labor2vec.modelo.pretrain import TransformerEncoder


def _modelo(semilla: int = 0) -> tuple[TransformerEncoder, dict]:
    torch.manual_seed(semilla)
    hp = hparams(HPARAMS_MINIMO)
    return TransformerEncoder(hp), lote_aleatorio(hp, n=4, largo=100, semilla=semilla)


def test_formas_de_salida():
    modelo, lote = _modelo()
    mlm, sop = modelo(lote)
    hp = modelo.hparams
    assert mlm.shape == (4, int(hp.mask_ratio * hp.max_length), hp.vocab_size)
    assert sop.shape == (4, hp.cls_num_targs)
    assert torch.isfinite(mlm).all() and torch.isfinite(sop).all()


def test_aprende_un_lote_fijo():
    """Sobreajuste intencional: la pérdida tiene que bajar claramente en un lote fijo."""
    modelo, lote = _modelo()
    opt = torch.optim.AdamW(modelo.parameters(), lr=3e-3)
    modelo.train()
    perdidas = []
    for _ in range(80):
        opt.zero_grad()
        mlm, sop = modelo(lote)
        perdida = modelo.calculate_total_loss(mlm, sop, lote)
        perdida.backward()
        opt.step()
        perdidas.append(perdida.item())
    assert perdidas[-1] < 0.8 * perdidas[0], perdidas


def test_relleno_no_afecta_a_los_tokens_validos():
    """Cambiar los tokens del relleno no debe cambiar la salida en las posiciones válidas."""
    modelo, lote = _modelo()
    modelo.eval()
    with torch.no_grad():
        y1 = modelo.transformer(lote["input_ids"], lote["padding_mask"])
        otro = {k: v.clone() for k, v in lote.items()}
        otro["input_ids"][:, 0, 100:] = 7
        y2 = modelo.transformer(otro["input_ids"], otro["padding_mask"])
    assert torch.allclose(y1[:, :100], y2[:, :100], atol=1e-5)


def test_cantidad_de_parametros_config_paper():
    """Con la configuración del paper el modelo debería tener ~8,4 M de parámetros (tabla A2)."""
    from labor2vec.modelo.config import HPARAMS_PAPER

    modelo = TransformerEncoder(hparams(HPARAMS_PAPER))
    n = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    assert 7e6 < n < 10e6, n
