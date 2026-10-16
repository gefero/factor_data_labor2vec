"""Línea de base del MLM: modelo unigrama con las frecuencias de los tokens de train."""

import numpy as np
import torch

from labor2vec.tokens.vocabulario import N_ESPECIALES


def logp_unigrama(dataset, vocab_size: int, n_docs: int = 20000, semilla: int = 0) -> torch.Tensor:
    """Log-probabilidad de cada token según su frecuencia en `n_docs` documentos de train (sin augmentación).

    Sólo se reparte masa entre los tokens no especiales (los únicos que se enmascaran), con suavizado de Laplace.
    """
    rng = np.random.default_rng(semilla)
    ks = rng.choice(len(dataset), size=min(n_docs, len(dataset)), replace=False)
    cuentas = np.zeros(vocab_size, dtype=np.float64)
    for k in ks:
        cuentas += np.bincount(dataset.documento(int(k)).tokens, minlength=vocab_size)
    cuentas[:N_ESPECIALES] = 0
    p = cuentas + 1.0
    p[:N_ESPECIALES] = 0
    p /= p.sum()
    with np.errstate(divide="ignore"):
        return torch.from_numpy(np.log(p)).float().clamp(min=-30.0)
