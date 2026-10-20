"""Clasificador binario para T1 (fine-tuning del transformer pre-entrenado, o GRU / transformer desde cero).

Replica lo verificado en el life2vec original (`cls_model.py`, `transformer_utils.py`; ver plans/PLAN_fase_6.md):
cabeza con *pooling* por atención, pérdida de entropía cruzada balanceada, RAdam con tasa de aprendizaje por capa,
`ExponentialLR` por época y embeddings de token congelados salvo algunos tokens especiales.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import numpy as np
from lightning.pytorch.utilities.parsing import AttributeDict

from labor2vec.modelo.gru import GRUEncoder
from labor2vec.modelo.transformer import Transformer
from labor2vec.modelo.transformer_utils import ScaleNorm, swish
from labor2vec.tareas import metricas
from labor2vec.tokens.vocabulario import CLS, SEP, UNK

# Ajustes por defecto del fine-tuning (conf/experiment/eos.yaml del original). Se pisan desde el YAML.
FT_DEFECTO = dict(
    tipo="transformer",            # transformer | gru
    desde_cero=False,              # transformer sin pre-entrenar
    congelar_embeddings=True,      # sólo se entrenan los embeddings de [CLS], [SEP] y [UNK]
    lr=0.01, decaimiento_capa=0.95, decaimiento_epoca=0.8, weight_decay_cabeza=0.01, weight_decay_encoder=0.0,
    dropout_cabeza=0.1, umbral=0.5,
)


class AttentionDecoder(nn.Module):
    """Cabeza del original: ff + Swish + ScaleNorm + AlphaDropout, *pooling* por atención (tanh y contexto aprendido,
    SigSoftmax enmascarado) y capa lineal a 2 clases."""

    def __init__(self, ancho: int, dropout: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.ff = nn.Linear(ancho, ancho)
        self.norm = ScaleNorm(ancho, eps)
        self.dropout = nn.AlphaDropout(dropout)
        self.pool = nn.Linear(ancho, ancho // 2)
        self.contexto = nn.Parameter(torch.randn(ancho // 2) * 0.05)
        self.salida = nn.Linear(ancho, 2)

    def forward(self, x, padding_mask):
        h = self.dropout(self.norm(swish(self.ff(x))))
        s = (torch.tanh(self.pool(h)) @ self.contexto).float()
        s = s.masked_fill(~padding_mask.bool(), float("-inf"))
        w = torch.softmax(s + torch.log(torch.sigmoid(s) + 1e-12), dim=-1)  # SigSoftmax
        return self.salida(torch.einsum("bs,bsh->bh", w.to(h.dtype), h))


def perdida_balanceada(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """0,5 · NLL media de los positivos + 0,5 · NLL media de los negativos (la «asimétrica PU» del código original)."""
    nll = F.cross_entropy(logits.float(), y, reduction="none")
    pos = y == 1
    partes = [nll[m].mean() for m in (pos, ~pos) if m.any()]
    return sum(partes) * (0.5 if len(partes) == 2 else 1.0)


class ClasificadorT1(pl.LightningModule):
    def __init__(self, hp_encoder: dict, ft: dict | None = None):
        super().__init__()
        self.save_hyperparameters({"hp_encoder": dict(hp_encoder), "ft": {**FT_DEFECTO, **(ft or {})}})
        hp, self.ft = AttributeDict(self.hparams.hp_encoder), AttributeDict(self.hparams.ft)
        if self.ft.tipo == "gru":
            self.encoder = GRUEncoder(hp.vocab_size, hp.n_tokens_fondo)
            ancho = self.encoder.ancho
        else:
            self.encoder = Transformer(hp)
            ancho = hp.hidden_size
        self.cabeza = AttentionDecoder(ancho, self.ft.dropout_cabeza)
        self._acum = []
        if self.ft.tipo == "transformer" and self.ft.congelar_embeddings:
            self._congelar_embeddings()

    def cargar_preentrenado(self, ruta) -> None:
        """Copia los pesos del encoder de un `modelo_final.pt` del pre-entrenamiento (descarta los decodificadores)."""
        ck = torch.load(ruta, map_location="cpu", weights_only=False)
        pesos = {k[len("transformer."):]: v for k, v in ck["state_dict"].items() if k.startswith("transformer.")}
        self.encoder.load_state_dict(pesos)

    def _congelar_embeddings(self) -> None:
        """Anula el gradiente de las filas de la matriz de tokens salvo [CLS], [SEP] y [UNK]."""
        tok = self.encoder.embedding.token
        param = tok.parametrizations.weight.original if hasattr(tok, "parametrizations") else tok.weight
        libres = torch.zeros(param.shape[0], 1)
        libres[[CLS, SEP, UNK]] = 1.0
        self.register_buffer("_filas_libres", libres, persistent=False)
        param.register_hook(lambda g: g * self._filas_libres.to(g.dtype))

    def forward(self, batch):
        x, mascara = batch["input_ids"].long(), batch["padding_mask"].long()
        return self.cabeza(self.encoder(x, mascara), mascara)

    def training_step(self, batch, batch_idx):
        perdida = perdida_balanceada(self(batch), batch["target"])
        self.log("train/loss", perdida, on_step=True, on_epoch=False, batch_size=len(batch["target"]))
        return perdida

    def _evaluar(self, batch):
        logits = self(batch).float()
        p = torch.softmax(logits, -1)[:, 1]
        self._acum.append((batch["target"].cpu(), p.cpu(), perdida_balanceada(logits, batch["target"]).cpu()))

    def validation_step(self, batch, batch_idx):
        self._evaluar(batch)

    def test_step(self, batch, batch_idx):
        self._evaluar(batch)

    def on_validation_epoch_start(self):
        self._acum = []

    on_test_epoch_start = on_validation_epoch_start

    def _fin(self, etapa: str):
        if not self._acum:
            return
        y = torch.cat([a[0] for a in self._acum]).numpy()
        p = torch.cat([a[1] for a in self._acum]).numpy()
        m = metricas.todas(y, p, self.ft.umbral)
        self.log_dict({f"{etapa}/{k}": float(v) for k, v in m.items()} | {
            f"{etapa}/loss": float(np.mean([float(a[2]) for a in self._acum]))})
        self._acum = []

    def on_validation_epoch_end(self):
        self._fin("val")

    def on_test_epoch_end(self):
        self._fin("test")

    def on_train_epoch_end(self):
        if self.ft.tipo == "transformer":
            self.encoder.redraw_projection_matrix(-1)  # nuevas proyecciones aleatorias del Performer

    @torch.no_grad()
    def predecir(self, loader, dispositivo=None) -> dict:
        """Probabilidades de positivo para todo el loader: {id, y, p} como arreglos numpy."""
        self.eval()
        dispositivo = dispositivo or next(self.parameters()).device
        ids, ys, ps = [], [], []
        for b in loader:
            b = {k: v.to(dispositivo) for k, v in b.items()}
            with torch.autocast(dispositivo.type, dtype=torch.bfloat16, enabled=dispositivo.type == "cuda"):
                logits = self(b)
            ps.append(torch.softmax(logits.float(), -1)[:, 1].cpu())
            ids.append(b["id"].cpu())
            ys.append(b["target"].cpu())
        return {"id": torch.cat(ids).numpy(), "y": torch.cat(ys).numpy(), "p": torch.cat(ps).numpy()}

    def configure_optimizers(self):
        ft = self.ft
        grupos = []

        def grupo(modulo, lr, wd):
            params = [p for p in modulo.parameters() if p.requires_grad]
            if params:
                grupos.append({"params": params, "lr": lr, "weight_decay": wd})

        grupo(self.cabeza, ft.lr, ft.weight_decay_cabeza)
        if ft.tipo == "transformer":
            n = len(self.encoder.encoders)
            grupo(self.encoder.embedding, ft.lr * ft.decaimiento_capa ** n, ft.weight_decay_encoder)
            for i, capa in enumerate(self.encoder.encoders):
                grupo(capa, ft.lr * ft.decaimiento_capa ** (n - i), ft.weight_decay_encoder)
        else:
            grupo(self.encoder, ft.lr, ft.weight_decay_encoder)
        opt = torch.optim.RAdam(grupos, betas=(0.9, 0.999), eps=1e-6)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": torch.optim.lr_scheduler.ExponentialLR(
            opt, ft.decaimiento_epoca), "interval": "epoch"}}
