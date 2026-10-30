# Copiado de carlomarxdk/life2vec-light (commit 9a0ef7a), src/models/pretrain.py
# Licencia MIT, (c) 2024 Germans Savcisens. Cambios: rutas de import, pytorch_lightning -> lightning.pytorch
# `sop_pesos` (pesos de clase de SOP), AUC de SOP y calendario OneCycle parametrizable (por defecto, los valores originales: 30 épocas x 375 pasos).

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from pathlib import Path
import logging

"""Custom code"""
from labor2vec.modelo.transformer_utils import ReZero
from labor2vec.modelo.transformer import Transformer, MaskedLanguageModel, SOP_Decoder

log = logging.getLogger(__name__)


class TransformerEncoder(pl.LightningModule):
    """Transformer with Masked Language Model and Sentence Order Prediction"""

    def __init__(self, hparams):
        super(TransformerEncoder, self).__init__()
        self.hparams.update(hparams)
        self.last_global_step = 0
        # 1. ENCODER
        self.transformer = Transformer(self.hparams)

        # 2. DECODER BLOCK
        self.task = self.hparams.training_task
        log.info("Training task: %s" % self.task)
        if "mlm" in self.task:
            # Number of outputs (for logging purposes)
            self.num_outputs = self.hparams.vocab_size

            # 2.1. DECODERS
            self.mlm_decoder = MaskedLanguageModel(
                self.hparams, self.transformer.embedding, act="tanh")
            self.sop_decoder = SOP_Decoder(self.hparams)
            # 2.2. LOSS
            # Weighting for the loss functions
            self.register_buffer("sop_weight", torch.tensor(0.2))
            self.register_buffer("mlm_weight", torch.tensor(0.8))
            # pesos de clase de SOP (por defecto, los de light; el repo completo de life2vec usa 1/0.9, 10, 10)
            pesos = self.hparams.get("sop_pesos", (1 / 0.8, 1 / 0.1, 1 / 0.1))
            self.register_buffer("sop_class_weight", torch.tensor([float(p) for p in pesos]))
            # Loss functions
            self.sop_loss = nn.CrossEntropyLoss(
                weight=self.sop_class_weight, label_smoothing=0.1)
            self.mlm_loss = nn.CrossEntropyLoss(ignore_index=0)
            # log-probabilidades unigrama (línea de base del MLM); se completan con `fijar_unigrama`
            self.register_buffer("logp_unigrama", torch.zeros(self.hparams.vocab_size))
            self.recall_sop = torchmetrics.classification.MulticlassRecall(
                num_classes=self.hparams.cls_num_targs, average=None)
            self.auc_sop = torchmetrics.classification.BinaryAUROC()
            self._acum = {}
        else:
            raise NotImplementedError()

    def forward(self, batch):
        """Forward pass that returns the logits for the masked language model and the sequence order prediction task."""
        # 1. ENCODER INPUT
        predicted = self.transformer(
            x=batch["input_ids"].long(),
            padding_mask=batch["padding_mask"].long()
        )
        # 2. MASKED LANGUAGE MODEL
        mlm_pred = self.mlm_decoder(predicted, batch)
        # 3. SEQUENCE ORDER PREDICTION Task
        # Embedding of the CLS token
        sop_pred = self.sop_decoder(predicted[:, 0])

        return mlm_pred, sop_pred

    def calculate_total_loss(self, mlm_preds, sop_preds, batch):
        return self.calcular_perdidas(mlm_preds, sop_preds, batch)["total"]

    def calcular_perdidas(self, mlm_preds, sop_preds, batch) -> dict:
        mlm_targs = batch["target_tokens"].long()
        sop_targs = batch["target_sop"].long()
        mlm_loss = self.mlm_loss(mlm_preds.permute(0, 2, 1), target=mlm_targs)
        sop_loss = self.sop_loss(sop_preds, target=sop_targs)
        total = self.sop_weight * sop_loss + self.mlm_weight * mlm_loss
        return {"total": total, "mlm": mlm_loss, "sop": sop_loss}

    def fijar_unigrama(self, logp: torch.Tensor) -> None:
        """Log-probabilidades unigrama sobre el vocabulario, para comparar la perplejidad del MLM."""
        self.logp_unigrama.copy_(logp)

    @torch.no_grad()
    def _metricas(self, mlm_preds, sop_preds, batch) -> dict:
        """Sumas sobre las posiciones enmascaradas del lote (el relleno tiene objetivo 0)."""
        targs = batch["target_tokens"].long()
        valido = targs != 0
        t = targs[valido]
        logits = mlm_preds[valido].float()
        nll = F.cross_entropy(logits, t, reduction="sum")
        return {
            "n": valido.sum().to(nll.dtype),
            "nll": nll,
            "nll_unigrama": -self.logp_unigrama[t].sum(),
            "aciertos": (logits.argmax(-1) == t).sum().to(nll.dtype),
            "sop_ok": (sop_preds.argmax(-1) == batch["target_sop"].long()).sum().to(nll.dtype),
            "n_sop": torch.tensor(float(len(sop_preds)), device=nll.device),
        }

    def training_step(self, batch, batch_idx):
        """Training Step"""
        mlm_preds, sop_preds = self(batch)
        perd = self.calcular_perdidas(mlm_preds, sop_preds, batch)
        m = self._metricas(mlm_preds, sop_preds, batch)
        n = len(batch["target_sop"])
        self.log_dict({"train/loss": perd["total"], "train/mlm_loss": perd["mlm"], "train/sop_loss": perd["sop"],
                       "train/mlm_acc": m["aciertos"] / m["n"].clamp(min=1),
                       "train/sop_acc": m["sop_ok"] / m["n_sop"]},
                      on_step=True, on_epoch=False, batch_size=n, prog_bar=False)
        return perd["total"]

    def _paso_evaluacion(self, batch, etapa: str):
        mlm_preds, sop_preds = self(batch)
        perd = self.calcular_perdidas(mlm_preds, sop_preds, batch)
        m = self._metricas(mlm_preds, sop_preds, batch)
        for k, v in m.items():
            self._acum[k] = self._acum.get(k, 0) + v
        self._acum["sop_loss"] = self._acum.get("sop_loss", 0) + perd["sop"].detach() * m["n_sop"]
        self.recall_sop.update(sop_preds.argmax(-1), batch["target_sop"].long())
        # AUC de «perturbado contra normal» con 1 - P(normal): no depende de ningún umbral
        self.auc_sop.update(1 - sop_preds.float().softmax(-1)[:, 0], (batch["target_sop"] != 0).long())
        return perd["total"]

    def _fin_evaluacion(self, etapa: str) -> None:
        a, n = self._acum, self._acum["n"].clamp(min=1)
        mlm = a["nll"] / n
        self.log_dict({f"{etapa}/mlm_loss": mlm, f"{etapa}/mlm_ppl": mlm.exp(),
                       f"{etapa}/mlm_acc": a["aciertos"] / n,
                       f"{etapa}/ppl_unigrama": (a["nll_unigrama"] / n).exp(),
                       f"{etapa}/sop_loss": a["sop_loss"] / a["n_sop"],
                       f"{etapa}/sop_acc": a["sop_ok"] / a["n_sop"]})
        for c, r in enumerate(self.recall_sop.compute()):
            self.log(f"{etapa}/sop_recall_c{c}", r)
        self.recall_sop.reset()
        self.log(f"{etapa}/sop_auc", self.auc_sop.compute())
        self.auc_sop.reset()
        self._acum = {}

    def on_train_epoch_end(self, *kwargs):
        """On Train Epoch End: Redraw the projection of the Attention-related matrices"""
        if self.hparams.attention_type == "performer":
            self.transformer.redraw_projection_matrix(-1)
        else:
            raise NotImplementedError(
                "We only have a Performer implementation.")

    def validation_step(self, batch, batch_idx):
        return self._paso_evaluacion(batch, "val")

    def on_validation_epoch_start(self):
        self._acum = {}
        self.recall_sop.reset()
        self.auc_sop.reset()

    def on_validation_epoch_end(self):
        if self._acum:  # no hay nada si Lightning hace la comprobación previa sin lotes
            self._fin_evaluacion("val")

    def test_step(self, batch, batch_idx):
        return self._paso_evaluacion(batch, "test")

    def on_test_epoch_start(self):
        self._acum = {}
        self.recall_sop.reset()
        self.auc_sop.reset()

    def on_test_epoch_end(self):
        if self._acum:
            self._fin_evaluacion("test")

    def configure_optimizers(self):
        """Configuration of the Optimizer and the Learning Rate Scheduler."""
        no_decay = [
            "bias",
            "norm",
            "age",
            "abspos",
            "token",
            "decoder.g"
        ]

        # It is advised to avoid the decay on the embedding weights, biases of the model and values of the ReZero gates.

        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            betas=(self.hparams.beta1, self.hparams.beta2),
            eps=self.hparams.epsilon,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                    optimizer, max_lr=self.hparams.learning_rate,
                    epochs=self.hparams.get("epochs", 30),
                    steps_per_epoch=self.hparams.get("steps_per_epoch", 375),
                    three_phase=False, pct_start=0.05, max_momentum=self.hparams.beta1,
                    div_factor=30
                ),
                "interval": "step",
                "frequency": 1,
                "name": "learning_rate",
            },
        }
