"""Carga de modelos pre-entrenados y de la tabla de conceptos (tokens) con su categoría y descripción."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
from lightning.pytorch.utilities.parsing import AttributeDict

from labor2vec.modelo.pretrain import TransformerEncoder
from labor2vec.tokens.vocabulario import N_ESPECIALES, SEXOS
from labor2vec.utiles.rutas import CORPUS, DESCRIPTORES, RELACIONES


def cargar_modelo(ruta) -> TransformerEncoder:
    """Reconstruye el modelo desde un `modelo_final.pt` ({state_dict, hparams}) escrito por `entrenar`."""
    ck = torch.load(ruta, map_location="cpu", weights_only=False)
    modelo = TransformerEncoder(AttributeDict(ck["hparams"]))
    modelo.load_state_dict(ck["state_dict"])
    return modelo.eval()


@torch.no_grad()
def matriz_embeddings(ruta) -> np.ndarray:
    """Matriz (vocab, dim) de embeddings de token de la entrada, centrada con la media de los tokens no especiales.

    La matriz que usa el modelo es `embedding.token.weight` con la parametrización de centrado aplicada (los tokens
    especiales quedan en cero); con *weight tying* la de salida es la misma. Las filas de los tokens especiales no se
    deben usar en ningún análisis (`N_ESPECIALES` primeras filas).
    """
    modelo = cargar_modelo(ruta)
    e = modelo.transformer.embedding.token.weight.detach().cpu().numpy().astype(np.float64)
    return e - e[N_ESPECIALES:].mean(0, keepdims=True)


@dataclass
class Conceptos:
    """Tabla de tokens no especiales (la fila i corresponde a la fila i + N_ESPECIALES de la matriz)."""

    tabla: pl.DataFrame  # token, id, categoria, frecuencia, valor, letra, r32, region, descripcion

    def __len__(self):
        return self.tabla.height

    def ids(self, categoria: str) -> np.ndarray:
        """Índices (en la matriz completa) de los tokens de una categoría."""
        return self.tabla.filter(pl.col("categoria") == categoria)["id"].to_numpy()

    def col(self, categoria: str, columna: str) -> np.ndarray:
        return self.tabla.filter(pl.col("categoria") == categoria)[columna].to_numpy()


def cargar_conceptos(corpus: Path = CORPUS, descriptores: Path = DESCRIPTORES, relaciones: Path = RELACIONES) -> Conceptos:
    """Tokens no especiales con categoría, valor ordinal (percentil, año, nivel), letra/r32 de las ramas, región de las
    provincias y la descripción del descriptor del MTEySS."""
    voc = pl.read_parquet(corpus / "vocabulario.parquet").sort("id").filter(pl.col("categoria") != "especial")
    ramas = pl.read_parquet(corpus / "ramas.parquet")
    # letra de cada r34: la más frecuente entre las relaciones
    letra_r34 = (pl.read_parquet(relaciones, columns=["r34", "letra"]).drop_nulls().group_by(["r34", "letra"]).len()
                 .sort(["r34", "len"], descending=[False, True]).unique("r34", keep="first").select("r34", "letra"))
    desc = {n: pl.read_parquet(descriptores / f"{n}.parquet")
            for n in ("r34", "r32", "provi", "letra", "tramo_empleo", "tr_anio_ini", "sexo")}

    def sufijo(t: str) -> int:
        return int(t.rsplit("_", 1)[1])

    filas = []
    rama_r34 = dict(zip(ramas["token"].to_list(), ramas["r34"].to_list()))
    rama_r32 = dict(zip(ramas["token"].to_list(), ramas["r32"].to_list()))
    letras = dict(zip(letra_r34["r34"].to_list(), letra_r34["letra"].to_list()))
    d_r34 = dict(zip(desc["r34"]["r34"].to_list(), desc["r34"]["descripcion"].to_list()))
    d_r32 = dict(zip(desc["r32"]["r32"].to_list(), desc["r32"]["descripcion"].to_list()))
    d_prov = {r["provi"]: (r["descripcion"], r["region"]) for r in desc["provi"].iter_rows(named=True)}
    d_tam = dict(zip(desc["tramo_empleo"]["tramo_empleo"].to_list(), desc["tramo_empleo"]["descripcion"].to_list()))
    d_ant = dict(zip(desc["tr_anio_ini"]["tr_anio_ini"].to_list(), desc["tr_anio_ini"]["descripcion"].to_list()))
    d_sexo = dict(zip(desc["sexo"]["sexo"].to_list(), desc["sexo"]["descripcion"].to_list()))
    sexo_de_token = {tok: cod for cod, tok in SEXOS.items()}
    r32_letra = {}  # letra de cada r32 (para los tokens RAMA2): la de sus r34
    for r34, l in letras.items():
        r32_letra.setdefault(int(r34) // 100, l)
    for t, i, cat, f in voc.iter_rows():
        valor = letra = r32 = region = None
        descripcion = ""
        if cat in ("salario", "nacimiento"):
            valor = sufijo(t)
        elif cat == "tamaño":
            valor = sufijo(t)
            descripcion = d_tam.get(valor, "")
        elif cat == "antigüedad":
            valor = sufijo(t)
            descripcion = d_ant.get(valor, "")
        elif cat == "sexo":
            valor = sexo_de_token[t]
            descripcion = d_sexo.get(valor, "")
        elif cat == "rama":
            r34 = sufijo(t)
            valor, r32, letra = r34, r34 // 100, letras.get(r34)
            descripcion = d_r34.get(r34, "")
        elif cat == "rama2":
            r32 = sufijo(t)
            valor, letra = r32, r32_letra.get(r32)
            descripcion = d_r32.get(r32, "")
        elif cat == "provincia":
            valor = sufijo(t)
            descripcion, region = d_prov.get(valor, ("", None))
        filas.append((t, i, cat, f, valor, letra, r32, region, descripcion))
    tabla = pl.DataFrame(filas, schema=["token", "id", "categoria", "frecuencia", "valor", "letra", "r32", "region",
                                        "descripcion"], orient="row",
                         schema_overrides={"valor": pl.Int64, "letra": pl.Int64, "r32": pl.Int64, "region": pl.String})
    assert tabla["id"].min() == N_ESPECIALES
    return Conceptos(tabla)
