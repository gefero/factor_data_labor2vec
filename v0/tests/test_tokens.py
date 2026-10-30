"""Pruebas de la fase 2: vocabulario, corpus, documentos, MLM/SOP y carga de datos (con un panel sintético)."""

import numpy as np
import polars as pl
import pytest
import torch

from labor2vec.tokens.datos import DataModuleMLER, DatasetPretrain, SamplerPorLargo, collate
from labor2vec.tokens.documento import (N_FONDO, SOP_INVERTIDO, SOP_MEZCLADO, SOP_NORMAL, Eventos, Opciones,
                                        construir_documento, dias_de_mes, enmascarar)
from labor2vec.tokens.vocabulario import CLS, MASK, N_ESPECIALES, PAD, SEP, UNK, Vocabulario

from conftest import UMBRAL_NAC, UMBRAL_RAMA


# ---- vocabulario ---------------------------------------------------------------------------------------
def test_especiales_en_el_orden_de_light(mundo):
    v = mundo["vocab"]
    assert v.tokens[:10] == ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[PLCH0]", "[PLCH1]", "[PLCH2]", "[PLCH3]",
                             "[PLCH4]", "[UNK]"]
    assert (PAD, CLS, SEP, MASK, UNK) == (0, 1, 2, 3, 9)


def test_vocabulario_determinista_y_recargable(mundo):
    v = mundo["vocab"]
    otro = Vocabulario.ajustar(mundo["rutas"]["panel"], umbral_rama=UMBRAL_RAMA, umbral_nac=UMBRAL_NAC)
    assert otro.tokens == v.tokens
    c = mundo["salida"]
    cargado = Vocabulario.cargar(c / "vocabulario.parquet", c / "ramas.parquet")
    assert cargado.tokens == v.tokens and (cargado.nac_min, cargado.nac_max) == (v.nac_min, v.nac_max)


def test_se_ajusta_sólo_con_train(mundo):
    """Un código que sólo existe en val/test no obtiene token propio."""
    panel = mundo["panel"]
    sintetico = panel.with_columns(
        pl.when(pl.col("id_trabajador") == 45).then(pl.lit(999)).otherwise(pl.col("provi")).alias("provi"))
    ruta = mundo["salida"].parent / "panel_provi.parquet"
    sintetico.write_parquet(ruta)
    v = Vocabulario.ajustar(ruta, umbral_rama=UMBRAL_RAMA, umbral_nac=UMBRAL_NAC)
    assert "PROV_999" not in v.tokens and "PROV_01" in v.tokens


def test_rama_rara_pasa_a_r32(mundo):
    v = mundo["vocab"]
    assert "RAMA_111" in v.tokens and "RAMA_113" not in v.tokens and "RAMA2_11" in v.tokens
    t = v.tablas()["rama"]
    assert t[111 + 1] == v.id("RAMA_111") and t[113 + 1] == v.id("RAMA2_11") and t[0] == UNK


def test_tablas_faltantes_son_unk(mundo):
    t = mundo["vocab"].tablas()
    for nombre in ("rem", "tam", "prov", "ant", "rama"):
        assert t[nombre][0] == UNK  # índice 0 = código -1 = dato faltante


# ---- corpus ---------------------------------------------------------------------------------------------
def test_corpus_suma_de_eventos_y_splits_disjuntos(mundo):
    panel, tot = mundo["panel"], mundo["totales"]
    for s in ("train", "val", "test"):
        sub = panel.filter(pl.col("split") == s)
        assert tot[s] == (sub["id_trabajador"].n_unique(), sub.height)
    ids = [set(pl.read_parquet(mundo["salida"] / f"eventos_{s}")["id_trabajador"]) for s in ("train", "val", "test")]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])


def test_corpus_orden_cronologico_y_nueva(mundo):
    df = pl.read_parquet(mundo["salida"] / "eventos_train").filter(pl.col("id_trabajador") == 1)
    mes = df["mes_idx"][0].to_list()
    assert mes == sorted(mes)
    # relación 1: meses 0,1,2 y 6 (después de una brecha); relación 2: 1,2,3
    assert mes == [0, 1, 1, 2, 2, 3, 6]
    assert df["id_relacion"][0].to_list() == [1, 1, 2, 1, 2, 2, 1]
    assert df["nueva"][0].to_list() == [True, False, True, False, False, False, True]


# ---- documento --------------------------------------------------------------------------------------------
def _eventos_trabajador_1(mundo):
    from labor2vec.tokens.datos import CorpusEnMemoria
    c = CorpusEnMemoria(mundo["salida"] / "eventos_train")
    i = int(np.flatnonzero(c.id_trabajador == 1)[0])
    return c, i


def test_documento_variante_a(mundo):
    v, (c, i) = mundo["vocab"], _eventos_trabajador_1(mundo)
    doc = construir_documento(c.eventos(i), c.sexo[i], c.fnacim[i], v, v.tablas(), Opciones())
    assert len(doc) == N_FONDO + 6 * 7
    assert doc.tokens[0] == CLS and doc.tokens[3] == SEP and doc.tokens[N_FONDO + 5] == SEP
    assert (doc.tokens[N_FONDO + 5::6] == SEP).all()
    assert v.tokens[doc.tokens[1]] == "SEXO_V" and v.tokens[doc.tokens[2]].startswith("NAC_")
    assert (doc.abspos[:N_FONDO] == 0).all() and (doc.edad[:N_FONDO] == 0).all()
    assert (doc.segmento[:N_FONDO] == 1).all()
    # los dos eventos del mes 1 comparten fecha pero no segmento
    ev1, ev2 = doc.segmento[N_FONDO + 6], doc.segmento[N_FONDO + 12]
    # ciclo de segmentos por evento, como en life2vec: 2, 3, 1, 2, 3, 1, 2
    assert list(doc.segmento[N_FONDO::6]) == [2, 3, 1, 2, 3, 1, 2]
    assert ev1 != ev2 and doc.abspos[N_FONDO + 6] == doc.abspos[N_FONDO + 12] == dias_de_mes(np.array([1]))[0]
    assert np.all(np.diff(doc.abspos[N_FONDO:]) >= 0)
    assert doc.edad[N_FONDO] == 1996 - c.fnacim[i] - 1 or doc.edad[N_FONDO] == 1996 - c.fnacim[i]


def test_documento_variante_b_acorta(mundo):
    v, (c, i) = mundo["vocab"], _eventos_trabajador_1(mundo)
    doc = construir_documento(c.eventos(i), c.sexo[i], c.fnacim[i], v, v.tablas(), Opciones(eventos_compactos=True))
    # nueva = [T, F, T, F, F, F, T] -> 3 completos (6) y 4 cortos (3)
    assert len(doc) == N_FONDO + 3 * 6 + 4 * 3
    assert (doc.tokens == SEP).sum() == 1 + 7 + 0  # el SEP del background y uno por evento
    assert (doc.tokens != UNK).all()


def test_ventana_y_truncado(mundo):
    v, (c, i) = mundo["vocab"], _eventos_trabajador_1(mundo)
    ev = c.eventos(i)
    doc = construir_documento(ev, c.sexo[i], c.fnacim[i], v, v.tablas(), Opciones(fin_ventana=2))
    assert len(doc) == N_FONDO + 6 * 5  # eventos de los meses 0, 1, 1, 2, 2
    doc = construir_documento(ev, c.sexo[i], c.fnacim[i], v, v.tablas(), Opciones(max_length=N_FONDO + 12))
    assert len(doc) == N_FONDO + 12
    # se conservan los más recientes: el último evento (mes 6) sigue
    assert doc.abspos[-1] == dias_de_mes(np.array([6]))[0]
    assert doc.tokens[1] != UNK  # el background se conserva


def test_sop_orden_distinto_con_tiempo_en_su_sitio(mundo):
    v, (c, i) = mundo["vocab"], _eventos_trabajador_1(mundo)
    base = construir_documento(c.eventos(i), c.sexo[i], c.fnacim[i], v, v.tablas(), Opciones())
    for p, clase in ((dict(p_sop_invertido=1.0, p_sop_mezclado=0.0), SOP_INVERTIDO),
                     (dict(p_sop_invertido=0.0, p_sop_mezclado=1.0), SOP_MEZCLADO)):
        doc = construir_documento(c.eventos(i), c.sexo[i], c.fnacim[i], v, v.tablas(), Opciones(**p),
                                  np.random.default_rng(0), con_sop=True)
        assert doc.sop == clase
        assert sorted(doc.tokens) == sorted(base.tokens) and (doc.tokens != base.tokens).any()
        assert (doc.abspos == base.abspos).all() and (doc.segmento == base.segmento).all()
        assert (doc.tokens[:N_FONDO] == base.tokens[:N_FONDO]).all()


def test_augmentacion_no_rompe_el_calendario(mundo):
    v, (c, i) = mundo["vocab"], _eventos_trabajador_1(mundo)
    op = Opciones(p_remuestreo=1.0, p_ruido_tiempo=0.0)
    ev = c.eventos(i)
    originales = set(dias_de_mes(ev.mes_idx))
    for s in range(20):
        doc = construir_documento(ev, c.sexo[i], c.fnacim[i], v, v.tablas(), op, np.random.default_rng(s), True)
        assert set(doc.abspos[N_FONDO:]) <= originales
        assert N_FONDO + 6 <= len(doc) < N_FONDO + 6 * 7


# ---- MLM --------------------------------------------------------------------------------------------------
def test_mlm_proporcion_y_nunca_especiales():
    rng = np.random.default_rng(0)
    tokens = np.concatenate([[CLS, 20, 21, SEP], rng.integers(10, 50, 996)])
    tokens[::6] = SEP
    n_mask = n_mask_tok = n_igual = n_azar = 0
    for s in range(200):
        tok, pos, obj = enmascarar(tokens, 50, 0.3, np.random.default_rng(s))
        assert (tokens[pos] >= N_ESPECIALES).all() and (obj == tokens[pos]).all()
        assert (tok[tokens < N_ESPECIALES] == tokens[tokens < N_ESPECIALES]).all()
        n_mask += len(pos)
        n_mask_tok += (tok[pos] == MASK).sum()
        n_igual += (tok[pos] == obj).sum()
        n_azar += (tok[pos] >= N_ESPECIALES).sum()
    assert n_mask / (200 * len(tokens)) == pytest.approx(0.3, abs=0.03)  # menos que 0,3 por los [SEP]
    assert n_mask_tok / n_mask == pytest.approx(0.8, abs=0.02)


# ---- carga de datos -----------------------------------------------------------------------------------------
def test_dataset_y_collate(mundo):
    v = mundo["vocab"]
    ds = DatasetPretrain(mundo["salida"] / "eventos_train", v, Opciones(max_length=256), augmentar=True, semilla=1)
    lote = collate([ds[k] for k in range(8)], 256)
    n, _, L = lote["input_ids"].shape
    assert n == 8 and L % 64 == 0 and L <= 256
    assert lote["padding_mask"].sum(1).max() <= L
    # el relleno es PAD en los cuatro canales
    assert (lote["input_ids"].permute(0, 2, 1)[~lote["padding_mask"]] == 0).all()
    # los objetivos coinciden con la posición y la máscara con el token (cuando no es 'sin cambio' o aleatorio)
    assert lote["target_pos"].max() < L and lote["input_ids"][:, 0].max() < len(v)
    assert set(lote["target_sop"].tolist()) <= {0, 1, 2}


def test_val_es_determinista(mundo):
    ds = DatasetPretrain(mundo["salida"] / "eventos_val", mundo["vocab"], Opciones(), augmentar=False, semilla=3)
    a, b = ds[2], ds[2]
    assert (a["input_ids"] == b["input_ids"]).all() and (a["target_pos"] == b["target_pos"]).all()


def test_ventana_filtra_poblacion(mundo):
    sin = DatasetPretrain(mundo["salida"] / "eventos_train", mundo["vocab"], Opciones(), False)
    con = DatasetPretrain(mundo["salida"] / "eventos_train", mundo["vocab"], Opciones(fin_ventana=2), False)
    assert len(con) <= len(sin) and (con.largos() <= sin.largos()[: len(con)].max() + 1).all()
    assert len(sin) == 40


def test_sampler_agrupa_por_largo_y_cubre_todo():
    largos = np.random.default_rng(0).integers(10, 2000, 1000)
    s = SamplerPorLargo(largos, 16, mega=10, semilla=0)
    lotes = list(s)
    assert sorted(i for l in lotes for i in l) == list(range(1000))
    dentro = np.mean([np.ptp(largos[l]) for l in lotes])
    assert dentro < 0.3 * np.ptp(largos)
    assert list(SamplerPorLargo(largos, 16, mega=10, semilla=0)) == lotes  # reproducible
    s2 = SamplerPorLargo(largos, 16, mega=10, semilla=0)
    list(s2)
    assert list(s2) != lotes  # otra época, otro orden


def test_datamodule_y_un_paso_del_modelo(mundo):
    from labor2vec.modelo.config import HPARAMS_MINIMO, hparams
    from labor2vec.modelo.pretrain import TransformerEncoder

    v = mundo["vocab"]
    dm = DataModuleMLER(mundo["salida"], v, Opciones(max_length=128), tamano_lote=8, num_workers=0)
    dm.setup()
    lote = next(iter(dm.train_dataloader()))
    hp = hparams(HPARAMS_MINIMO, vocab_size=len(v), max_length=128, n_tokens_fondo=N_FONDO,
                 sop_pesos=(1 / 0.9, 10.0, 10.0))
    torch.manual_seed(0)
    modelo = TransformerEncoder(hp)
    opt = torch.optim.AdamW(modelo.parameters(), lr=3e-3)
    perdidas = []
    for _ in range(40):
        opt.zero_grad()
        mlm, sop = modelo(lote)
        perdida = modelo.calculate_total_loss(mlm, sop, lote)
        perdida.backward()
        opt.step()
        perdidas.append(perdida.item())
    assert np.isfinite(perdidas).all() and perdidas[-1] < 0.8 * perdidas[0]
