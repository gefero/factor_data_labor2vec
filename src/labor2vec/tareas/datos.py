"""Dataset, sampler balanceado y DataModule para ajustar T1 (plans/PLAN_fase_6.md, sección 6.2).

Reutiliza el documento del pre-entrenamiento (`DatasetPretrain.documento`) sin SOP ni MLM, restringido a la cohorte de
`tareas.t1`. Las etiquetas se leen con pyarrow: este módulo no importa polars (los nodos de Mendieta no lo soportan).
"""

from functools import partial
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset

from labor2vec.tokens.datos import MULTIPLO, DatasetPretrain, SamplerPorLargo
from labor2vec.tokens.documento import Opciones
from labor2vec.tokens.vocabulario import Vocabulario
from labor2vec.utiles.rutas import CORPUS, T1


def leer_etiquetas(ruta=T1, etiqueta: str = "y_2019") -> dict[int, tuple[int, int]]:
    """{id_trabajador: (edad_corte, etiqueta)} de la cohorte de T1."""
    t = pq.read_table(ruta, columns=["id_trabajador", "edad_corte", etiqueta])
    return dict(zip(t["id_trabajador"].to_pylist(), zip(t["edad_corte"].to_pylist(), t[etiqueta].to_pylist())))


class DatasetT1(Dataset):
    """Un documento por trabajador de la cohorte, con su etiqueta (1 = sale del empleo registrado)."""

    def __init__(self, directorio, vocab: Vocabulario, op: Opciones, augmentar: bool, ruta_t1=T1,
                 etiqueta: str = "y_2019", semilla: int = 0):
        self.base = DatasetPretrain(directorio, vocab, op, augmentar, semilla)
        etiquetas = leer_etiquetas(ruta_t1, etiqueta)
        ids = self.base.corpus.id_trabajador[self.base.indices]
        en_cohorte = np.array([int(i) in etiquetas for i in ids], dtype=bool)
        self.base.indices = self.base.indices[en_cohorte]  # largos() y documento() usan estos índices
        self.id = self.base.corpus.id_trabajador[self.base.indices].astype(np.int64)
        self.y = np.array([etiquetas[int(i)][1] for i in self.id], dtype=np.int64)
        self.edad = np.array([etiquetas[int(i)][0] for i in self.id], dtype=np.int64)
        self.sexo = self.base.corpus.sexo[self.base.indices].astype(np.int64)

    def __len__(self):
        return len(self.base.indices)

    def largos(self) -> np.ndarray:
        return self.base.largos()

    def __getitem__(self, k: int) -> dict:
        rng = self.base._generador(k)
        doc = self.base.documento(k, rng, self.base.augmentar, con_sop=False)
        return {"input_ids": doc.como_matriz(), "target": self.y[k], "id": self.id[k]}


def collate(muestras: list[dict], max_length: int, multiplo: int = MULTIPLO) -> dict:
    largo = max(m["input_ids"].shape[1] for m in muestras)
    L = min(-(-largo // multiplo) * multiplo, max_length)
    ids = np.zeros((len(muestras), 4, L), dtype=np.int64)
    padding = np.zeros((len(muestras), L), dtype=bool)
    for j, m in enumerate(muestras):
        k = m["input_ids"].shape[1]
        ids[j, :, :k] = m["input_ids"]
        padding[j, :k] = True
    return {"input_ids": torch.from_numpy(ids), "padding_mask": torch.from_numpy(padding),
            "target": torch.tensor([m["target"] for m in muestras], dtype=torch.int64),
            "id": torch.tensor([m["id"] for m in muestras], dtype=torch.int64)}


class SamplerBalanceado(torch.utils.data.Sampler):
    """Lotes con ~50 % de positivos: se sortean los índices de la pasada con reposición, con probabilidad inversa a la
    frecuencia de su clase, y se agrupan por largo como `SamplerPorLargo`. Igual que el `WeightedRandomSampler` del
    life2vec original."""

    def __init__(self, largos: np.ndarray, y: np.ndarray, tamano: int, mega: int = 50, semilla: int = 0):
        self.largos, self.tamano, self.mega, self.semilla, self.epoca = np.asarray(largos), tamano, mega, semilla, 0
        y = np.asarray(y)
        frec = np.bincount(y, minlength=2).astype(float)
        w = (1.0 / np.maximum(frec, 1))[y]
        self.p = w / w.sum()

    def set_epoch(self, epoca: int) -> None:
        self.epoca = epoca

    def __iter__(self):
        n = len(self.largos)
        rng = np.random.default_rng([self.semilla, self.epoca])
        orden = rng.choice(n, size=n, replace=True, p=self.p)
        lotes, paso = [], self.tamano * self.mega
        for a in range(0, n, paso):
            grupo = orden[a: a + paso]
            grupo = grupo[np.argsort(self.largos[grupo], kind="stable")]
            lotes += [grupo[b: b + self.tamano].tolist() for b in range(0, len(grupo), self.tamano)]
        if lotes and len(lotes[-1]) < self.tamano:
            lotes.pop()
        self.epoca += 1
        yield from (lotes[i] for i in rng.permutation(len(lotes)))

    def __len__(self):
        return len(self.largos) // self.tamano


class DataModuleT1(pl.LightningDataModule):
    def __init__(self, corpus=CORPUS, vocab: Vocabulario | None = None, op: Opciones | None = None, ruta_t1=T1,
                 etiqueta: str = "y_2019", tamano_lote: int = 32, num_workers: int = 4, semilla: int = 0,
                 epoca_inicial: int = 0):
        super().__init__()
        self.corpus, self.op, self.ruta_t1, self.etiqueta = Path(corpus), op or Opciones(), ruta_t1, etiqueta
        self.vocab = vocab or Vocabulario.cargar(self.corpus / "vocabulario.parquet", self.corpus / "ramas.parquet")
        self.tamano, self.num_workers, self.semilla, self.epoca_inicial = tamano_lote, num_workers, semilla, epoca_inicial

    def setup(self, stage=None):
        self.ds = {s: DatasetT1(self.corpus / f"eventos_{s}", self.vocab, self.op, augmentar=(s == "train"),
                                ruta_t1=self.ruta_t1, etiqueta=self.etiqueta, semilla=self.semilla)
                   for s in ("train", "val", "test") if (self.corpus / f"eventos_{s}").exists()}

    def _loader(self, split: str, entrenar: bool) -> DataLoader:
        ds = self.ds[split]
        if entrenar:
            sampler = SamplerBalanceado(ds.largos(), ds.y, self.tamano, semilla=self.semilla)
            sampler.set_epoch(self.epoca_inicial)
        else:  # val y test completos, en orden fijo y con la prevalencia real
            sampler = SamplerPorLargo(ds.largos(), self.tamano, mezclar=False, epoca_fija=True)
        return DataLoader(ds, batch_sampler=sampler, num_workers=self.num_workers,
                          collate_fn=partial(collate, max_length=self.op.max_length),
                          persistent_workers=self.num_workers > 0, pin_memory=torch.cuda.is_available())

    def train_dataloader(self):
        return self._loader("train", True)

    def val_dataloader(self):
        return self._loader("val", False)

    def test_dataloader(self):
        return self._loader("test", False)
