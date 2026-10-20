"""Pruebas de la fase 5 con embeddings sintéticos de estructura conocida."""

import numpy as np
import pytest
import torch

from labor2vec.analisis import metricas as m
from labor2vec.analisis.embeddings import cargar_modelo, matriz_embeddings
from labor2vec.modelo.config import HPARAMS_MINIMO, hparams
from labor2vec.modelo.pretrain import TransformerEncoder
from labor2vec.tokens.vocabulario import N_ESPECIALES


def _linea(n=60, d=16, ruido=0.02, semilla=0):
    """n puntos sobre un arco de 90 grados en un plano de d dimensiones: el orden del índice es el del espacio, y la
    distancia coseno entre dos puntos crece con la distancia entre sus índices."""
    rng = np.random.default_rng(semilla)
    q, _ = np.linalg.qr(rng.normal(size=(d, 2)))
    theta = np.linspace(0, np.pi / 2, n)[:, None]
    return np.cos(theta) * q[:, 0] + np.sin(theta) * q[:, 1] + ruido * rng.normal(size=(n, d))


def test_orden_de_una_curva_vs_ruido():
    e, v = _linea(), np.arange(60)
    r = m.orden_pc1(e, v, n_perm=200)
    assert r["rho"] > 0.95 and r["p"] < 0.01
    c = m.consistencia_ordinal(e, v, k=4, ventana=3, n_perm=200)
    assert c["prop"] > 0.9 and c["p"] < 0.01
    ruido = np.random.default_rng(1).normal(size=(60, 16))
    assert m.consistencia_ordinal(ruido, v, 4, 3, n_perm=200)["p"] > 0.05
    assert m.orden_pc1(ruido, v, n_perm=200)["p"] > 0.01


def test_orden_por_distancias_en_curva_y_en_ruido():
    """La medida por pares no supone que el orden sea una recta: sube en una curva en herradura y es ~0 en ruido."""
    t = np.linspace(0, 1, 60)
    theta = t * 1.2 * np.pi
    e = np.stack([np.cos(theta), np.sin(theta)], 1) @ np.random.default_rng(0).normal(size=(2, 16))
    e = e + 0.01 * np.random.default_rng(1).normal(size=e.shape)
    v = np.arange(60)
    r = m.orden_distancias(e, v, n_perm=50)
    assert r["rho"] > 0.8 and r["p"] < 0.05 and abs(r["nulo"]) < 0.1
    ruido = np.random.default_rng(2).normal(size=(60, 16))
    assert abs(m.orden_distancias(ruido, v, n_perm=50)["rho"]) < 0.2


def test_pureza_de_grupos_separados():
    rng = np.random.default_rng(0)
    centros = rng.normal(size=(4, 16)) * 3
    et = np.repeat(np.arange(4), 20)
    e = centros[et] + rng.normal(size=(80, 16))
    r = m.pureza(e, et, k=5, n_perm=200)
    assert r["pureza"] > 0.95 and r["nulo"] == pytest.approx(0.25, abs=0.05) and r["p"] < 0.01
    assert m.pureza(rng.normal(size=(80, 16)), et, 5, 200)["p"] > 0.05


def test_orden_pares_en_pocos_niveles():
    assert m.orden_pares(_linea(n=4, ruido=0.0))["rho"] > 0.9  # los empates entre pares a igual distancia ordinal bajan el máximo


def test_robustez_entre_modelos_y_nulos():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(60, 16))
    d1 = m.distancias_coseno(base)
    d2 = m.distancias_coseno(base + 0.3 * rng.normal(size=base.shape))  # otro modelo: misma estructura con ruido
    perm = m.distancias_coseno(m.nulo_columnas_permutadas(base, rng))
    gauss = m.distancias_coseno(m.nulo_gaussiano(base, rng))
    assert m.rho_modelos(d1, d2) > 0.7
    assert abs(m.rho_modelos(d1, perm)) < 0.2 and abs(m.rho_modelos(d1, gauss)) < 0.2
    assert m.test_permutacion(d1, d2, n_iter=300)["p"] < 0.01
    assert m.test_permutacion(d1, gauss, n_iter=300)["p"] > 0.05


def test_test_de_permutacion_coincide_con_spearman():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(40, 8))
    d1, d2 = m.distancias_coseno(a), m.distancias_coseno(a + 0.5 * rng.normal(size=a.shape))
    assert m.test_permutacion(d1, d2, n_iter=50)["rho"] == pytest.approx(m.rho_modelos(d1, d2), abs=1e-9)


def test_fdr_bh():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    aj = m.fdr_bh(p)
    assert aj == pytest.approx([0.02, 0.04, 0.04, 0.02])  # procedimiento de Benjamini-Hochberg
    assert (aj >= p).all() and (aj <= 1).all()


def test_hubness_cuenta_vecinos():
    e = np.random.default_rng(0).normal(size=(50, 8))
    g = m.grado_entrada(e, k=5)
    assert g.sum() == 50 * 5  # cada token aporta exactamente k aristas
    r = m.resumen_hubness(g, {"a": np.arange(25), "b": np.arange(25, 50)})
    assert r["por_grupo"]["a"]["n"] == 25 and 0 <= r["prop_cero"] <= 1
    # un token muy cercano a muchos otros es un hub: asimetría positiva
    h = np.random.default_rng(1).normal(size=(60, 8))
    h[:, 0] += 5
    h[0] = np.array([5] + [0] * 7)
    assert m.resumen_hubness(m.grado_entrada(h, 5), {})["asimetria"] > 0.5


def test_el_cargador_extrae_la_matriz_del_modelo(tmp_path):
    hp = hparams(HPARAMS_MINIMO, vocab_size=50, n_tokens_fondo=4)
    torch.manual_seed(0)
    modelo = TransformerEncoder(hp)
    ruta = tmp_path / "modelo_final.pt"
    torch.save({"state_dict": modelo.state_dict(), "hparams": dict(modelo.hparams)}, ruta)
    e = matriz_embeddings(ruta)
    assert e.shape == (50, hp.hidden_size)
    esperado = modelo.transformer.embedding.token.weight.detach().numpy()
    resto = esperado[N_ESPECIALES:]
    assert np.allclose(e[N_ESPECIALES:], resto - resto.mean(0), atol=1e-6)
    assert np.allclose(e[N_ESPECIALES:].mean(0), 0, atol=1e-6)
    assert isinstance(cargar_modelo(ruta), TransformerEncoder)
