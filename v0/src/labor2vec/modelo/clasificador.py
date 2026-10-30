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
    congelar_encoder=False,        # sonda lineal: se entrena sólo la cabeza y el encoder queda fijo y en eval
    lr=0.01, decaimiento_capa=0.95, decaimiento_epoca=0.8, weight_decay_cabeza=0.01, weight_decay_encoder=0.0,
    dropout_cabeza=0.1, umbral=0.5,
    n_clases=2,                    # T1 y T3: 2; T2: 5 quintiles; T4: 3 clases ordenadas
    perdida="balanceada",          # balanceada | ordinal (CDW + focal + suavizado, como el original)
)


class AttentionDecoder(nn.Module):
    """Cabeza del original: ff + Swish + ScaleNorm + AlphaDropout, *pooling* por atención (tanh y contexto aprendido,
    SigSoftmax enmascarado) y capa lineal a `n_clases`."""

    def __init__(self, ancho: int, dropout: float = 0.1, eps: float = 1e-6, n_clases: int = 2):
        super().__init__()
        self.ff = nn.Linear(ancho, ancho)
        self.norm = ScaleNorm(ancho, eps)
        self.dropout = nn.AlphaDropout(dropout)
        self.pool = nn.Linear(ancho, ancho // 2)
        self.contexto = nn.Parameter(torch.randn(ancho // 2) * 0.05)
        self.salida = nn.Linear(ancho, n_clases)

    def resumen(self, x, padding_mask):
        """Vector de la persona: el *pooling* por atención, justo antes de la capa lineal de salida."""
        h = self.dropout(self.norm(swish(self.ff(x))))
        s = (torch.tanh(self.pool(h)) @ self.contexto).float()
        s = s.masked_fill(~padding_mask.bool(), float("-inf"))
        w = torch.softmax(s + torch.log(torch.sigmoid(s) + 1e-12), dim=-1)  # SigSoftmax
        return torch.einsum("bs,bsh->bh", w.to(h.dtype), h)

    def forward(self, x, padding_mask):
        return self.salida(self.resumen(x, padding_mask))


def perdida_balanceada(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Promedio, entre las clases presentes en el lote, de la NLL media de cada clase. Con 2 clases es 0,5 · NLL media de
    los positivos + 0,5 · NLL media de los negativos (la «asimétrica PU» del código original)."""
    nll = F.cross_entropy(logits.float(), y, reduction="none")
    partes = [nll[y == c].mean() for c in torch.unique(y)]
    return sum(partes) / len(partes)


def cdw_ce(logits: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Entropía cruzada ponderada por la distancia entre clases (CDW-CE, `CDW_CELoss` del original con transformación
    logarítmica): -Σ_i log(1 - p_i) · log(1 + |i - y|)^alpha, que castiga más que el modelo ponga masa lejos de la clase
    verdadera. La clase verdadera pesa 0."""
    p = torch.softmax(logits.float(), -1)
    clases = torch.arange(p.shape[-1], device=p.device)
    w = torch.log1p((clases[None, :] - y[:, None]).abs().float()) ** alpha
    return -(torch.log(1 - p + 1e-5) * w).sum(-1).mean()


def focal(logits: torch.Tensor, y: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    """Pérdida focal: -(1 - p_y)^gamma · log p_y."""
    logp = torch.log_softmax(logits.float(), -1).gather(1, y[:, None])[:, 0]
    return (-((1 - logp.exp()) ** gamma) * logp).mean()


def perdida_ordinal(logits: torch.Tensor, y: torch.Tensor, pesos=(0.3, 1.0, 0.1), gamma: float = 2.0) -> torch.Tensor:
    """0,3 · CDW-CE + 1 · focal + 0,1 · suavizado de etiquetas (entropía cruzada contra la distribución uniforme).

    Son los pesos fijos del paper; el código original los combina con una ponderación adaptativa (CoV-weighting) y no se
    pudo verificar cuál produjo sus resultados. Se usa softmax común (el original aplica SigSoftmax)."""
    suavizado = -torch.log_softmax(logits.float(), -1).mean(-1).mean()
    return pesos[0] * cdw_ce(logits, y) + pesos[1] * focal(logits, y, gamma) + pesos[2] * suavizado


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
        self.cabeza = AttentionDecoder(ancho, self.ft.dropout_cabeza, n_clases=self.ft.n_clases)
        self._acum = []
        if self.ft.tipo == "transformer" and self.ft.congelar_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        elif self.ft.tipo == "transformer" and self.ft.congelar_embeddings:
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

    def train(self, mode: bool = True):
        super().train(mode)
        if self.ft.tipo == "transformer" and self.ft.congelar_encoder:
            self.encoder.eval()  # sin dropout en el encoder fijo
        return self

    def forward(self, batch):
        x, mascara = batch["input_ids"].long(), batch["padding_mask"].long()
        return self.cabeza(self.encoder(x, mascara), mascara)

    def _perdida(self, logits, y):
        return perdida_ordinal(logits, y) if self.ft.perdida == "ordinal" else perdida_balanceada(logits, y)

    def training_step(self, batch, batch_idx):
        perdida = self._perdida(self(batch), batch["target"])
        self.log("train/loss", perdida, on_step=True, on_epoch=False, batch_size=len(batch["target"]))
        return perdida

    def _evaluar(self, batch):
        logits = self(batch).float()
        self._acum.append((batch["target"].cpu(), torch.softmax(logits, -1).cpu(),
                           self._perdida(logits, batch["target"]).cpu()))

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
        probs = torch.cat([a[1] for a in self._acum]).numpy()
        m = metricas.multiclase(y, probs) if self.ft.n_clases > 2 else metricas.todas(y, probs[:, 1], self.ft.umbral)
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
        """Probabilidades para todo el loader: {id, y, p} como arreglos numpy. `p` es la probabilidad de la clase 1 con
        2 clases, o la matriz (n, n_clases) con más."""
        self.eval()
        dispositivo = dispositivo or next(self.parameters()).device
        ids, ys, ps = [], [], []
        for b in loader:
            b = {k: v.to(dispositivo) for k, v in b.items()}
            with torch.autocast(dispositivo.type, dtype=torch.bfloat16, enabled=dispositivo.type == "cuda"):
                logits = self(b)
            probs = torch.softmax(logits.float(), -1)
            ps.append((probs[:, 1] if self.ft.n_clases == 2 else probs).cpu())
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
