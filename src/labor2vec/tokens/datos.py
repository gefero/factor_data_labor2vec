"""Dataset, sampler por largo y DataModule para el pre-entrenamiento (MLM + SOP)."""

from functools import partial
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info

from labor2vec.tokens.documento import N_FONDO, Eventos, Opciones, construir_documento, enmascarar
from labor2vec.tokens.vocabulario import Vocabulario
from labor2vec.utiles.rutas import CORPUS

MULTIPLO = 64  # el largo del lote se redondea hacia arriba a un múltiplo de esto (atención local)
CAMPOS = ("mes_idx", "pct_rem", "tramo", "r34", "provi", "antig", "nueva")


class CorpusEnMemoria:
    """Eventos de un split como arreglos planos con offsets por trabajador."""

    def __init__(self, directorio: Path):
        archivos = sorted(Path(directorio).glob("*.parquet"))
        if not archivos:
            raise FileNotFoundError(f"no hay corpus en {directorio}")
        tabla = pq.ParquetDataset(archivos).read(columns=["id_trabajador", "sexo", "fnacim", *CAMPOS]).combine_chunks()
        self.id_trabajador = tabla["id_trabajador"].to_numpy()
        self.sexo = tabla["sexo"].to_numpy()
        self.fnacim = tabla["fnacim"].to_numpy()
        lista = tabla["mes_idx"].chunk(0)
        self.offsets = lista.offsets.to_numpy().astype(np.int64)
        self.col = {c: tabla[c].chunk(0).values.to_numpy(zero_copy_only=False) for c in CAMPOS}

    def __len__(self):
        return len(self.id_trabajador)

    def eventos(self, i: int, hasta: int | None = None) -> Eventos:
        a, b = self.offsets[i], self.offsets[i + 1] if hasta is None else hasta
        return Eventos(*(self.col[c][a:b] for c in CAMPOS))


class DatasetPretrain(Dataset):
    """Un documento por trabajador, con augmentación, SOP y MLM. Devuelve arreglos numpy."""

    def __init__(self, directorio, vocab: Vocabulario, op: Opciones, augmentar: bool, semilla: int = 0):
        self.corpus = CorpusEnMemoria(directorio)
        self.vocab, self.tablas, self.op = vocab, vocab.tablas(), op
        self.augmentar, self.semilla = augmentar, semilla
        self._rng, self._rng_semilla = None, None
        # eventos dentro de la ventana (están ordenados por mes: son un prefijo de cada trabajador)
        mes, off = self.corpus.col["mes_idx"], self.corpus.offsets
        dentro = np.concatenate([[0], np.cumsum(mes <= (op.fin_ventana if op.fin_ventana is not None else 10**6))])
        self.fin = off[:-1] + (dentro[off[1:]] - dentro[off[:-1]])
        self.indices = np.flatnonzero(self.fin > off[:-1])  # con al menos un evento en la ventana

    def __len__(self):
        return len(self.indices)

    def largos(self) -> np.ndarray:
        """Largo aproximado (tokens) de cada documento antes de la augmentación y con el tope de max_length."""
        off = self.corpus.offsets
        n = (self.fin - off[:-1])[self.indices]
        if not self.op.eventos_compactos:
            tok = 6 * n
        else:
            nueva = np.concatenate([[0], np.cumsum(self.corpus.col["nueva"][: off[-1]])])
            completos = nueva[self.fin] - nueva[off[:-1]]
            tok = 3 * n + 3 * completos[self.indices]
        return np.minimum(N_FONDO + tok, self.op.max_length)

    def _generador(self, k: int) -> np.random.Generator:
        if not self.augmentar:  # val/test: determinista por trabajador
            return np.random.default_rng([self.semilla, k])
        info = get_worker_info()
        semilla = info.seed if info is not None else self.semilla
        if self._rng is None or self._rng_semilla != semilla:
            self._rng, self._rng_semilla = np.random.default_rng(semilla), semilla
        return self._rng

    def documento(self, k: int, rng: np.random.Generator | None = None, augmentar: bool = False,
                  con_sop: bool = False):
        """Documento del trabajador `k` (sin enmascarar)."""
        i = self.indices[k]
        c = self.corpus
        return construir_documento(c.eventos(i, self.fin[i]), c.sexo[i], c.fnacim[i], self.vocab, self.tablas,
                                   self.op, rng, augmentar=augmentar, con_sop=con_sop)

    def __getitem__(self, k: int) -> dict:
        rng = self._generador(k)
        doc = self.documento(k, rng, self.augmentar, con_sop=True)
        tokens, pos, obj = enmascarar(doc.tokens, len(self.vocab), self.op.mask_ratio, rng)
        doc.tokens = tokens
        return {"input_ids": doc.como_matriz(), "target_pos": pos, "target_tokens": obj, "target_sop": doc.sop}


def collate(muestras: list[dict], max_length: int, multiplo: int = MULTIPLO, mask_ratio: float = 0.3) -> dict:
    """Rellena hasta el largo máximo del lote (redondeado a `multiplo`, sin pasar de `max_length`)."""
    largo = max(m["input_ids"].shape[1] for m in muestras)
    L = min(-(-largo // multiplo) * multiplo, max_length)
    n_mask = max(1, int(np.floor(mask_ratio * L)))
    n = len(muestras)
    ids = np.zeros((n, 4, L), dtype=np.int64)
    padding = np.zeros((n, L), dtype=bool)
    pos = np.full((n, n_mask), L - 1, dtype=np.int64)  # se rellena con la última posición y objetivo 0 (ignorado)
    obj = np.zeros((n, n_mask), dtype=np.int64)
    for j, m in enumerate(muestras):
        k = m["input_ids"].shape[1]
        ids[j, :, :k] = m["input_ids"]
        padding[j, :k] = True
        p = m["target_pos"][:n_mask]
        pos[j, : len(p)] = p
        obj[j, : len(p)] = m["target_tokens"][: len(p)]
    return {"input_ids": torch.from_numpy(ids), "padding_mask": torch.from_numpy(padding),
            "target_pos": torch.from_numpy(pos), "target_tokens": torch.from_numpy(obj),
            "target_sop": torch.tensor([m["target_sop"] for m in muestras], dtype=torch.int64)}


class SamplerPorLargo(torch.utils.data.Sampler):
    """Lotes de secuencias de largo parecido: se mezclan los índices, se arman mega-lotes de `mega` x `tamano`,
    se ordenan por largo dentro de cada uno y se parten en lotes, que se mezclan de nuevo."""

    def __init__(self, largos: np.ndarray, tamano: int, mega: int = 50, mezclar: bool = True, semilla: int = 0,
                 descartar_ultimo: bool = False, epoca_fija: bool = False):
        self.largos, self.tamano, self.mega = np.asarray(largos), tamano, mega
        self.mezclar, self.semilla, self.descartar = mezclar, semilla, descartar_ultimo
        self.epoca, self.epoca_fija = 0, epoca_fija  # epoca_fija: mismo orden en cada pasada (validación)

    def set_epoch(self, epoca: int) -> None:
        self.epoca = epoca

    def __iter__(self):
        n = len(self.largos)
        rng = np.random.default_rng([self.semilla, self.epoca])
        orden = rng.permutation(n) if self.mezclar else np.arange(n)
        lotes = []
        paso = self.tamano * self.mega
        for a in range(0, n, paso):
            grupo = orden[a: a + paso]
            grupo = grupo[np.argsort(self.largos[grupo], kind="stable")]
            lotes += [grupo[b: b + self.tamano].tolist() for b in range(0, len(grupo), self.tamano)]
        if self.descartar and lotes and len(lotes[-1]) < self.tamano:
            lotes.pop()
        if self.mezclar:
            lotes = [lotes[i] for i in rng.permutation(len(lotes))]
        if not self.epoca_fija:
            self.epoca += 1
        yield from lotes

    def __len__(self):
        n = len(self.largos)
        return n // self.tamano if self.descartar else -(-n // self.tamano)


class DataModuleMLER(pl.LightningDataModule):
    def __init__(self, corpus=CORPUS, vocab: Vocabulario | None = None, op: Opciones | None = None,
                 tamano_lote: int = 32, num_workers: int = 4, semilla: int = 0, epoca_inicial: int = 0):
        super().__init__()
        self.corpus, self.op = Path(corpus), op or Opciones()
        self.vocab = vocab or Vocabulario.cargar(self.corpus / "vocabulario.parquet", self.corpus / "ramas.parquet")
        self.tamano, self.num_workers, self.semilla = tamano_lote, num_workers, semilla
        self.epoca_inicial = epoca_inicial  # para reanudar con el orden de lotes que correspondía

    def setup(self, stage=None):
        self.ds = {s: DatasetPretrain(self.corpus / f"eventos_{s}", self.vocab, self.op, augmentar=(s == "train"),
                                      semilla=self.semilla) for s in ("train", "val", "test")
                   if (self.corpus / f"eventos_{s}").exists()}

    def _loader(self, split: str, entrenar: bool) -> DataLoader:
        ds = self.ds[split]
        # validación y test: orden mezclado pero fijo, así un subconjunto de lotes es siempre el mismo y representativo
        sampler = SamplerPorLargo(ds.largos(), self.tamano, mezclar=True, semilla=self.semilla,
                                  descartar_ultimo=entrenar, epoca_fija=not entrenar)
        if entrenar:
            sampler.set_epoch(self.epoca_inicial)
        return DataLoader(ds, batch_sampler=sampler, num_workers=self.num_workers,
                          collate_fn=partial(collate, max_length=self.op.max_length, mask_ratio=self.op.mask_ratio),
                          persistent_workers=self.num_workers > 0, pin_memory=torch.cuda.is_available())

    def train_dataloader(self):
        return self._loader("train", True)

    def val_dataloader(self):
        return self._loader("val", False)

    def test_dataloader(self):
        return self._loader("test", False)
