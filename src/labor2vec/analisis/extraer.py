"""Fase 7, paso en el cluster: vectores de persona, gradientes para TCAV y saliency (SmoothGrad) de un modelo ajustado.

    python -m labor2vec.analisis.extraer --config configs/ajuste_t1.yaml --modelo runs/t1_l2v_lr0001/modelo_final.pt \\
        --salida runs/fase7/t1 [--n-saliency 2000] [--n-copias 25] [--ruido 0.15]
    python -m labor2vec.analisis.extraer --config configs/ajuste_t1.yaml --modelo runs/pre_150ep/modelo_final.pt \\
        --sin-cabeza --salida runs/fase7/pre

Escribe en la carpeta de salida:
- `personas.npz`: para val y test (`val_*`, `test_*`): `id`, `y`, `logit` (logit de la clase 1 menos el de la 0), `vec`
  (vector de persona: salida del *pooling* de la cabeza; sin cabeza, el promedio de la última capa), `m_ult` y `m_pen`
  (promedio sobre los tokens reales de la salida de la última y de la penúltima capa del encoder). Sólo en test y con
  cabeza: `g_ult` y `g_pen`, la suma sobre tokens del gradiente del logit respecto de esas capas. La derivada del logit al
  sumar `eps·v` al estado de todos los tokens de una persona es `g · v`, así que con `g` guardado cualquier vector de
  concepto `v` se evalúa después sin el modelo (TCAV).
- `saliency.npz` (sólo con cabeza): muestra del test estratificada por `y` y por quintil del logit, con `token`, `abspos`
  y `puntaje` de SmoothGrad por token (todos concatenados; `largo` da el largo de cada persona).

Sin polars ni scipy: tiene que correr en los nodos de Mendieta.
"""

import argparse
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from labor2vec.entrenar import _ruta, cargar_config
from labor2vec.modelo.clasificador import ClasificadorT1
from labor2vec.tareas.datos import DataModuleT1, collate
from labor2vec.tokens.documento import Opciones, opciones_de
from labor2vec.tokens.vocabulario import N_ESPECIALES


class Sonda:
    """Pasa lotes por el clasificador y captura, con *hooks*, lo que hace falta sin tocar el modelo.

    - `encoder.embedding.token`: la salida se reemplaza por una copia que requiere gradiente y, con `sigma > 0`, lleva ruido
      gaussiano (SmoothGrad). Como los parámetros están congelados, el gradiente sólo se calcula hacia las activaciones.
    - `encoder.encoders[-1]` y `[-2]`: salidas de las dos últimas capas (para promedios y para `g`).
    El modelo debe estar en `eval()` (sin dropout) y en fp32.
    """

    def __init__(self, modelo: ClasificadorT1, sigma: float = 0.0):
        """`sigma`: desvío del ruido de SmoothGrad; el ruido sólo se aplica dentro de `smoothgrad`."""
        self.m, self.sigma_sg, self.sigma, self.est = modelo, sigma, 0.0, {}
        for p in modelo.parameters():
            p.requires_grad_(False)
        enc = modelo.encoder
        enc.embedding.token.register_forward_hook(self._hook_token)
        enc.encoders[-2].register_forward_hook(lambda mod, i, out: self.est.__setitem__("pen", out))

    def _hook_token(self, mod, inp, out):
        e = out.detach()
        y = e + self.sigma * torch.randn_like(e) if self.sigma else e.clone()
        y.requires_grad_(torch.is_grad_enabled())
        self.est["emb"], self.est["emb_ruido"] = e, y
        return y

    def pasar(self, lote: dict, con_cabeza: bool = True, con_grad: bool = False) -> dict:
        """vec, logit, promedios por capa y (con `con_grad`) `g_ult`, `g_pen`."""
        x, mascara = lote["input_ids"].long(), lote["padding_mask"].long()
        m = mascara.unsqueeze(-1).to(torch.float32)
        with torch.set_grad_enabled(con_grad):
            h = self.m.encoder(x, mascara)
            pen = self.est["pen"]
            media = lambda t: (t * m).sum(1) / m.sum(1)  # noqa: E731
            sal = {"m_ult": media(h), "m_pen": media(pen)}
            if con_cabeza:
                vec = self.m.cabeza.resumen(h, mascara)
                lg = self.m.cabeza.salida(vec)
                sal |= {"vec": vec, "logit": lg[:, 1] - lg[:, 0]}
            else:
                sal |= {"vec": sal["m_ult"]}
            if con_grad:
                g_ult, g_pen = torch.autograd.grad(sal["logit"].sum(), [h, pen], retain_graph=True)
                sal |= {"g_ult": (g_ult * m).sum(1), "g_pen": (g_pen * m).sum(1)}
        return sal

    def smoothgrad(self, lote: dict, n_copias: int) -> torch.Tensor:
        """Puntaje por token (B, S): |gradiente medio del logit respecto del embedding · embedding|, normalizado a suma 1
        por persona. El gradiente se promedia sobre `n_copias` pasadas con ruido en los embeddings de token."""
        x, mascara = lote["input_ids"].long(), lote["padding_mask"].long()
        acum, self.sigma = 0, self.sigma_sg
        for _ in range(n_copias):
            with torch.enable_grad():
                h = self.m.encoder(x, mascara)
                lg = self.m.cabeza(h, mascara)
                acum = acum + torch.autograd.grad((lg[:, 1] - lg[:, 0]).sum(), self.est["emb_ruido"])[0]
        self.sigma = 0.0
        s = (acum / n_copias * self.est["emb"]).sum(-1).abs() * mascara
        return s / s.sum(1, keepdim=True).clamp_min(1e-12)


def sigma_ruido(modelo: ClasificadorT1, fraccion: float) -> float:
    """`fraccion` × el rango de valores de los embeddings de token (sin los especiales)."""
    with torch.no_grad():
        w = modelo.encoder.embedding.token.weight[N_ESPECIALES:]
    return float(fraccion * (w.max() - w.min()))


def a_cpu(sal: dict) -> dict:
    return {k: v.detach().cpu().numpy().astype(np.float32) for k, v in sal.items()}


def recorrer(sonda: Sonda, loader, dispositivo, con_cabeza: bool, con_grad: bool) -> dict:
    partes: dict[str, list] = {}
    for lote in loader:
        lote = {k: v.to(dispositivo) for k, v in lote.items()}
        sal = a_cpu(sonda.pasar(lote, con_cabeza, con_grad))
        sal |= {"id": lote["id"].cpu().numpy(), "y": lote["target"].cpu().numpy()}
        for k, v in sal.items():
            partes.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in partes.items()}


def muestra_estratificada(y: np.ndarray, logit: np.ndarray, n: int, semilla: int = 0) -> np.ndarray:
    """Índices de ~n personas: la misma cantidad por cada celda (y × quintil del logit), o todas si la celda es más chica."""
    rng = np.random.default_rng(semilla)
    q = np.searchsorted(np.quantile(logit, [0.2, 0.4, 0.6, 0.8]), logit)
    celdas = [np.flatnonzero((y == a) & (q == b)) for a in np.unique(y) for b in range(5)]
    por = max(1, n // len(celdas))
    return np.sort(np.concatenate([rng.choice(c, min(por, len(c)), replace=False) for c in celdas if len(c)]))


def saliency(sonda: Sonda, ds, indices: np.ndarray, op: Opciones, dispositivo, n_copias: int, lote: int) -> dict:
    """SmoothGrad de las personas `indices` del dataset (posiciones dentro de `ds`)."""
    largos = ds.largos()[indices]
    orden = indices[np.argsort(largos, kind="stable")]
    lotes = [orden[a: a + lote].tolist() for a in range(0, len(orden), lote)]
    loader = DataLoader(ds, batch_sampler=lotes, collate_fn=partial(collate, max_length=op.max_length))
    sal: dict[str, list] = {"id": [], "y": [], "largo": [], "token": [], "abspos": [], "puntaje": []}
    for b in loader:
        b = {k: v.to(dispositivo) for k, v in b.items()}
        s = sonda.smoothgrad(b, n_copias).cpu().numpy()
        for j in range(len(s)):
            k = int(b["padding_mask"][j].sum())
            sal["id"].append(int(b["id"][j]))
            sal["y"].append(int(b["target"][j]))
            sal["largo"].append(k)
            sal["token"].append(b["input_ids"][j, 0, :k].cpu().numpy().astype(np.int16))
            sal["abspos"].append(b["input_ids"][j, 1, :k].cpu().numpy().astype(np.int32))
            sal["puntaje"].append(s[j, :k].astype(np.float32))
    return {k: np.concatenate(v) if k in ("token", "abspos", "puntaje") else np.array(v) for k, v in sal.items()}


def cargar_modelo(ruta, cfg: dict, sin_cabeza: bool) -> ClasificadorT1:
    ck = torch.load(Path(ruta).expanduser(), map_location="cpu", weights_only=False)
    if sin_cabeza:  # encoder pre-entrenado: sus hparams son los del encoder, la cabeza queda sin usar
        m = ClasificadorT1(ck["hparams"])
        m.cargar_preentrenado(ruta)
    else:
        m = ClasificadorT1(ck["hparams"]["hp_encoder"], ck["hparams"]["ft"])
        m.load_state_dict(ck["state_dict"])
    return m.eval()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--set", action="append", default=[], dest="cambios", metavar="CLAVE=VALOR")
    ap.add_argument("--sin-cabeza", action="store_true", help="encoder pre-entrenado: sin logit, g ni saliency")
    ap.add_argument("--n-saliency", type=int, default=2000)
    ap.add_argument("--n-copias", type=int, default=25)
    ap.add_argument("--ruido", type=float, default=0.15, help="σ de SmoothGrad como fracción del rango de los embeddings")
    ap.add_argument("--lote-grad", type=int, default=8, help="lote de la pasada de test con gradientes (con 32 no entra en 24 GB)")
    ap.add_argument("--lote-saliency", type=int, default=8)
    ap.add_argument("--max-personas", type=int, help="sólo para pruebas: limita val y test")
    args = ap.parse_args(argv)

    cfg = cargar_config(_ruta(args.config), args.cambios)
    d = cfg["datos"]
    op = opciones_de(d)
    dm = DataModuleT1(_ruta(d["corpus"]), None, op, _ruta(d["t1"]), d["etiqueta"], d["tamano_lote"], d["num_workers"],
                      cfg["semilla"])
    dm.setup()
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = cargar_modelo(args.modelo, cfg, args.sin_cabeza).to(dispositivo)
    sonda = Sonda(modelo, sigma_ruido(modelo, args.ruido))
    salida = _ruta(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    out = {}
    for split in ("val", "test"):
        if split == "test" and not args.sin_cabeza:
            dm.tamano = args.lote_grad  # los gradientes de todas las capas en fp32 ocupan mucha más memoria que val
        loader = getattr(dm, f"{split}_dataloader")()
        if args.max_personas:
            loader = (b for _, b in zip(range(-(-args.max_personas // dm.tamano)), loader))
        r = recorrer(sonda, loader, dispositivo, not args.sin_cabeza, con_grad=(split == "test" and not args.sin_cabeza))
        out |= {f"{split}_{k}": v for k, v in r.items()}
        print(f"{split}: {len(r['id'])} personas, {time.time() - t0:.0f} s", flush=True)
    np.savez_compressed(salida / "personas.npz", **out)

    if not args.sin_cabeza and args.n_saliency:
        pos = {int(i): k for k, i in enumerate(dm.ds["test"].id)}
        elegidas = muestra_estratificada(out["test_y"], out["test_logit"], args.n_saliency)
        indices = np.array([pos[int(out["test_id"][k])] for k in elegidas])
        sal = saliency(sonda, dm.ds["test"], indices, op, dispositivo, args.n_copias, args.lote_saliency)
        np.savez_compressed(salida / "saliency.npz", **sal)
        print(f"saliency: {len(indices)} personas, {time.time() - t0:.0f} s", flush=True)
    print(f"listo; {salida}")


if __name__ == "__main__":
    main()
