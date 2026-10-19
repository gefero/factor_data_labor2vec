"""Pre-entrenamiento (MLM + SOP) del backbone sobre el corpus de la MLER.

    python -m labor2vec.entrenar --config configs/preentrenamiento.yaml [--set a.b=valor ...]
                                 [--reanudar] [--max-minutos N] [--max-pasos N] [--salida DIR]

Con `--reanudar` retoma desde `last.ckpt` de la carpeta de salida. `--max-minutos` detiene el entrenamiento a tiempo
(y guarda `last.ckpt`) para encadenar jobs de Slurm: volver a encolar el mismo comando.

`pasos_por_epoca` y `--max-pasos` se miden en pasos del optimizador; el lote efectivo es `tamano_lote` x `acumular`.
Al terminar se guarda `modelo_final.pt` (sólo pesos). Si la pérdida deja de ser finita se corta con un error y
`last.ckpt` conserva el último estado bueno.
"""

import argparse
import copy
import json
import time
from pathlib import Path

import lightning.pytorch as pl
import torch
import yaml
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from labor2vec.modelo.config import HPARAMS_PAPER, hparams
from labor2vec.modelo.pretrain import TransformerEncoder
from labor2vec.tokens.datos import DataModuleMLER
from labor2vec.tokens.documento import N_FONDO, Opciones
from labor2vec.tokens.linea_base import logp_unigrama
from labor2vec.tokens.vocabulario import Vocabulario
from labor2vec.utiles.rutas import RAIZ


def _en_paso_del_optimizador(trainer, batch_idx: int) -> bool:
    """Verdadero cuando el micro-lote que acaba de terminar cerró un paso del optimizador (acumulación de gradientes)."""
    return (batch_idx + 1) % trainer.accumulate_grad_batches == 0


class ParadaSegura(pl.Callback):
    """Corta el entrenamiento al llegar al tiempo o al paso indicado, guardando `last.ckpt` desde dentro del bucle.

    Guardar después de `fit` registra la época ya incrementada, y al reanudar Lightning daría por terminada una época
    a medias; guardado acá, el checkpoint es consistente con los de `every_n_train_steps`. Sólo corta entre pasos del
    optimizador, para no perder gradientes acumulados a medias.
    """

    def __init__(self, ruta: Path, max_segundos: float | None = None, max_pasos: int | None = None):
        self.ruta, self.max_pasos = ruta, max_pasos
        self.limite = time.time() + max_segundos if max_segundos else None
        self.paro = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not _en_paso_del_optimizador(trainer, batch_idx):
            return
        if (self.limite and time.time() >= self.limite) or (self.max_pasos and trainer.global_step >= self.max_pasos):
            self.paro = True
            trainer.save_checkpoint(self.ruta)
            trainer.should_stop = True


class GuardiaNaN(pl.Callback):
    """Corta con un error si la pérdida deja de ser finita (no se pisa `last.ckpt`: queda el último estado bueno)."""

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        perdida = outputs["loss"] if isinstance(outputs, dict) else outputs
        if perdida is not None and not torch.isfinite(perdida).all():
            raise FloatingPointError(f"pérdida no finita en el paso {trainer.global_step}")


class Rendimiento(pl.Callback):
    """Registra secuencias/s, tokens/s (con relleno) y memoria pico de la GPU cada `cada` pasos del optimizador."""

    def __init__(self, cada: int):
        self.cada, self.t, self.seqs, self.toks, self.paso = cada, None, 0, 0, 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        n, _, L = batch["input_ids"].shape
        self.seqs, self.toks = self.seqs + n, self.toks + n * L
        if not _en_paso_del_optimizador(trainer, batch_idx):
            return
        if self.t is None:  # el primer intervalo incluye el arranque (cuDNN, caché): se descarta
            self.t, self.seqs, self.toks, self.paso = time.time(), 0, 0, trainer.global_step
            return
        if trainer.global_step % self.cada == 0:
            dt = time.time() - self.t
            m = {"perf/seq_s": self.seqs / dt, "perf/tokens_s": self.toks / dt,
                 "perf/pasos_s": (trainer.global_step - self.paso) / dt}
            if torch.cuda.is_available():
                m["perf/mem_pico_gib"] = torch.cuda.max_memory_allocated() / 2**30
            pl_module.log_dict(m, on_step=True, on_epoch=False, batch_size=n)
            self.t, self.seqs, self.toks, self.paso = time.time(), 0, 0, trainer.global_step


def _ruta(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else RAIZ / p


def cargar_config(ruta, cambios: list[str] | None = None) -> dict:
    cfg = yaml.safe_load(Path(ruta).read_text())
    for c in cambios or []:
        clave, valor = c.split("=", 1)
        *camino, hoja = clave.split(".")
        d = cfg
        for k in camino:
            d = d.setdefault(k, {})
        d[hoja] = yaml.safe_load(valor)
    return cfg


def preparar(cfg: dict, epoca_inicial: int = 0):
    """Devuelve (datamodule, modelo) a partir de la configuración."""
    d, m, e = cfg["datos"], copy.deepcopy(cfg.get("modelo", {})), cfg["entrenamiento"]
    corpus = _ruta(d["corpus"])
    vocab = Vocabulario.cargar(corpus / "vocabulario.parquet", corpus / "ramas.parquet")
    mask_ratio = m.get("mask_ratio", HPARAMS_PAPER["mask_ratio"])
    op = Opciones(max_length=d["max_length"], fin_ventana=d["fin_ventana"], eventos_compactos=d["eventos_compactos"],
                  mask_ratio=mask_ratio)
    dm = DataModuleMLER(corpus, vocab, op, d["tamano_lote"], d["num_workers"], cfg["semilla"], epoca_inicial)
    dm.setup()

    acumular = e.get("acumular", 1)
    pasos = e.get("pasos_por_epoca") or -(-len(dm.train_dataloader()) // acumular)  # pasos del optimizador
    hp = hparams(HPARAMS_PAPER, vocab_size=len(vocab), max_length=d["max_length"], n_tokens_fondo=N_FONDO,
                 epochs=e["epocas"], steps_per_epoch=pasos, **m)
    hp.sop_pesos = tuple(hp.sop_pesos)
    modelo = TransformerEncoder(hp)
    modelo.fijar_unigrama(logp_unigrama(dm.ds["train"], len(vocab), e["muestra_unigrama"], cfg["semilla"]))
    return dm, modelo


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/preentrenamiento.yaml")
    ap.add_argument("--set", action="append", default=[], dest="cambios", metavar="CLAVE=VALOR")
    ap.add_argument("--reanudar", action="store_true")
    ap.add_argument("--max-minutos", type=float)
    ap.add_argument("--max-pasos", type=int, help="detiene el entrenamiento en este paso (guarda last.ckpt)")
    ap.add_argument("--salida")
    args = ap.parse_args(argv)

    cfg = cargar_config(_ruta(args.config), args.cambios)
    e = cfg["entrenamiento"]
    salida = _ruta(args.salida or e["salida"])
    salida.mkdir(parents=True, exist_ok=True)
    last = salida / "last.ckpt"
    retomar = args.reanudar and last.exists()
    epoca_inicial = 0
    if retomar:
        epoca_inicial = torch.load(last, map_location="cpu", weights_only=False)["epoch"]
    pl.seed_everything(cfg["semilla"], workers=True)
    dm, modelo = preparar(cfg, epoca_inicial)
    (salida / "config.json").write_text(json.dumps({**cfg, "hparams_modelo": dict(modelo.hparams)}, indent=2,
                                                   default=str))

    ckpts = [
        ModelCheckpoint(salida, save_last=True, every_n_train_steps=e["guardar_cada_pasos"], save_top_k=0),
        ModelCheckpoint(salida, filename="mejor", monitor="val/mlm_loss", mode="min", save_top_k=1,
                        save_weights_only=True),
    ]
    parada = ParadaSegura(last, args.max_minutos * 60 if args.max_minutos else None, args.max_pasos)
    acumular = e.get("acumular", 1)
    val_cada_pasos = e.get("val_cada_pasos")  # alternativa a val_cada_epocas, en pasos del optimizador
    trainer = pl.Trainer(
        max_epochs=e["epocas"], precision=e["precision"], accelerator="auto", devices=1,
        accumulate_grad_batches=acumular,
        limit_train_batches=e["pasos_por_epoca"] * acumular if e.get("pasos_por_epoca") else 1.0,
        limit_val_batches=e.get("val_lotes") or 1.0,
        val_check_interval=val_cada_pasos * acumular if val_cada_pasos else 1.0,
        check_val_every_n_epoch=e.get("val_cada_epocas", 1),
        gradient_clip_val=e.get("clip_gradiente"), log_every_n_steps=e["log_cada_pasos"],
        callbacks=[*ckpts, parada, GuardiaNaN(), Rendimiento(e["log_cada_pasos"]), LearningRateMonitor("step")],
        logger=CSVLogger(salida, name="logs"), enable_progress_bar=False, num_sanity_val_steps=0,
        default_root_dir=salida)
    t0 = time.time()
    trainer.fit(modelo, datamodule=dm, ckpt_path=str(last) if retomar else None)
    if not parada.paro:
        trainer.save_checkpoint(last)  # fin normal: queda el estado final (el corte anticipado ya se guardó)
        torch.save({"state_dict": modelo.state_dict(), "hparams": dict(modelo.hparams)}, salida / "modelo_final.pt")
    print(f"listo en {time.time() - t0:.0f} s; época {trainer.current_epoch}, paso {trainer.global_step}; {salida}")


if __name__ == "__main__":
    main()
