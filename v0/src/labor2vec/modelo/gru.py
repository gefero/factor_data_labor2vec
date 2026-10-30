"""GRU bidireccional de línea de base: mismos embeddings que el transformer (token, posición, edad y segmento), sin
pre-entrenar. Replica `SimpleGRU` del life2vec original (3 capas, 256 de ancho en total, dropout 0,27)."""

import torch.nn as nn
from lightning.pytorch.utilities.parsing import AttributeDict

from labor2vec.modelo.embeddings import Embeddings


class GRUEncoder(nn.Module):
    """Interfaz igual a `Transformer.forward(x, padding_mask)`: devuelve (B, L, ancho)."""

    def __init__(self, vocab_size: int, n_tokens_fondo: int, ancho: int = 256, capas: int = 3, dropout: float = 0.27):
        super().__init__()
        hp = AttributeDict(hidden_size=ancho, vocab_size=vocab_size, parametrize_emb=True, norm_input_emb=False,
                           emb_dropout=dropout, n_tokens_fondo=n_tokens_fondo)
        self.embedding = Embeddings(hp)
        self.gru = nn.GRU(ancho, ancho // 2, num_layers=capas, batch_first=True, bidirectional=True,
                          dropout=dropout if capas > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.ancho = ancho

    def forward(self, x, padding_mask):
        e, _ = self.embedding(tokens=x[:, 0], position=x[:, 1], age=x[:, 2], segment=x[:, 3])
        largos = padding_mask.long().sum(1).clamp(min=1).cpu()
        empaquetado = nn.utils.rnn.pack_padded_sequence(e, largos, batch_first=True, enforce_sorted=False)
        salida, _ = self.gru(empaquetado)
        salida, _ = nn.utils.rnn.pad_packed_sequence(salida, batch_first=True, total_length=e.shape[1])
        return self.dropout(salida)
