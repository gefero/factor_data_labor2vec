"""Hiperparámetros del backbone y lotes sintéticos para pruebas."""

import numpy as np
import torch
from lightning.pytorch.utilities.parsing import AttributeDict

# Configuración final de life2vec: SocialComplexityLab/life2vec, conf/experiment/pretrain.yaml
HPARAMS_PAPER = dict(
    vocab_size=2043,
    max_length=2560,
    hidden_size=280,
    hidden_ff=2210,
    hidden_act="swish",
    n_encoders=5,
    n_heads=10,
    n_local=7,
    local_window_size=38,
    norm_type="rezero",
    att_dropout=0.1,
    fw_dropout=0.1,
    dc_dropout=0.1,
    emb_dropout=0.1,
    parametrize_emb=True,
    norm_input_emb=False,
    norm_output_emb=True,
    weight_tying="wt",
    training_task="mlm",
    attention_type="performer",
    num_random_features=436,
    learning_rate=5e-3,
    weight_decay=0.01,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-6,
    cls_num_targs=3,
    mask_ratio=0.30,
    epochs=30,
    steps_per_epoch=375,
)

# Configuración mínima para pruebas en CPU
HPARAMS_MINIMO = dict(
    HPARAMS_PAPER,
    vocab_size=50,
    max_length=128,
    hidden_size=32,
    hidden_ff=64,
    n_encoders=2,
    n_heads=4,
    n_local=2,
    local_window_size=8,
    num_random_features=16,
)


def hparams(base: dict = HPARAMS_PAPER, **cambios) -> AttributeDict:
    """Devuelve una copia de `base` con los `cambios`, con acceso por atributo."""
    desconocidos = set(cambios) - set(base)
    if desconocidos:
        raise KeyError(f"hiperparámetros desconocidos: {sorted(desconocidos)}")
    return AttributeDict({**base, **cambios})


def lote_aleatorio(hp: AttributeDict, n: int, largo: int | None = None, semilla: int = 0) -> dict:
    """Lote sintético con el formato que espera el modelo de pre-entrenamiento.

    `input_ids` tiene forma (n, 4, max_length): tokens, posición absoluta, edad y segmento.
    Los tokens 0-9 se reservan como especiales (0 = [PAD]); las secuencias se rellenan con 0
    a partir de `largo`.
    """
    rng = np.random.default_rng(semilla)
    L = hp.max_length
    largo = L if largo is None else largo
    n_mask = int(np.floor(hp.mask_ratio * L))

    ids = np.zeros((n, 4, L), dtype=np.int64)
    ids[:, 0, :largo] = rng.integers(10, hp.vocab_size, size=(n, largo))
    ids[:, 1, :largo] = np.sort(rng.integers(0, 9500, size=(n, largo)), axis=1)  # días desde 1996
    ids[:, 2, :largo] = rng.integers(18, 70, size=(n, 1))                        # edad
    ids[:, 3, :largo] = rng.integers(1, 3, size=(n, largo))                      # segmento
    padding = np.zeros((n, L), dtype=bool)
    padding[:, :largo] = True

    # posiciones a predecir; el resto se rellena con la última posición y objetivo 0 (ignorado)
    pos = np.full((n, n_mask), L - 1, dtype=np.int64)
    obj = np.zeros((n, n_mask), dtype=np.int64)
    k = int(np.floor(hp.mask_ratio * largo))
    for i in range(n):
        p = rng.choice(np.arange(1, largo), size=k, replace=False)
        pos[i, :k] = p
        obj[i, :k] = ids[i, 0, p]
        ids[i, 0, p] = 3  # token de máscara sintético

    return {
        "input_ids": torch.from_numpy(ids),
        "padding_mask": torch.from_numpy(padding),
        "target_pos": torch.from_numpy(pos),
        "target_tokens": torch.from_numpy(obj),
        "target_sop": torch.from_numpy(rng.integers(0, 3, size=n)),
    }
