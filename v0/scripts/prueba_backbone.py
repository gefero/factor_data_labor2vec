"""Mide el backbone de life2vec con la configuración del paper: memoria pico y pasos por segundo.

Corre en CPU (versión reducida, para verificar el script) o en GPU (la usa scripts/cluster/prueba_mendieta.sbatch):

    python scripts/prueba_backbone.py                          # CPU, lote 2, largo 512, 3 pasos
    python scripts/prueba_backbone.py --dispositivo cuda --lotes 8 16 --pasos 30

Imprime la hora de cada etapa sin buffer; si algo se cuelga, faulthandler vuelca la pila cada 60 s.
"""

import argparse
import faulthandler
import time

faulthandler.dump_traceback_later(60, repeat=True)
T0 = time.time()


def paso(msg: str) -> None:
    print(f"{time.time() - T0:7.1f}s  {msg}", flush=True)


def medir(modelo, hp, lote_n: int, largo: int, pasos: int, dispositivo, precision: str) -> None:
    import torch

    from labor2vec.modelo.config import lote_aleatorio

    lote = {k: v.to(dispositivo) for k, v in lote_aleatorio(hp, n=lote_n, largo=largo).items()}
    opt = torch.optim.AdamW(modelo.parameters(), lr=1e-4)
    autocast = torch.autocast(dispositivo.type, dtype=torch.bfloat16, enabled=precision == "bf16")
    es_cuda = dispositivo.type == "cuda"
    if es_cuda:
        torch.cuda.reset_peak_memory_stats()

    def un_paso():
        opt.zero_grad(set_to_none=True)
        with autocast:
            mlm, sop = modelo(lote)
            perdida = modelo.calculate_total_loss(mlm.float(), sop.float(), lote)
        perdida.backward()
        opt.step()
        return perdida.item()

    try:
        perdida = un_paso()  # calentamiento
        if es_cuda:
            torch.cuda.synchronize()
        t = time.time()
        for _ in range(pasos):
            perdida = un_paso()
        if es_cuda:
            torch.cuda.synchronize()
        pps = pasos / (time.time() - t)
        mem = f", memoria pico {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB" if es_cuda else ""
        paso(f"lote {lote_n:>3} x {largo} tokens, {precision}: {pps:.2f} pasos/s, "
             f"{pps * lote_n:.1f} secuencias/s{mem}, pérdida {perdida:.3f}")
    except torch.OutOfMemoryError:
        paso(f"lote {lote_n:>3} x {largo} tokens, {precision}: sin memoria")
        if es_cuda:
            torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dispositivo", default="cpu")
    ap.add_argument("--lotes", type=int, nargs="+", default=[2])
    ap.add_argument("--largo", type=int, default=None, help="tokens válidos por secuencia (por defecto, max_length)")
    ap.add_argument("--max-length", type=int, default=None)
    ap.add_argument("--pasos", type=int, default=3)
    ap.add_argument("--precision", nargs="+", default=["fp32"], choices=["fp32", "bf16"])
    args = ap.parse_args()

    paso("inicio")
    import torch

    paso(f"torch {torch.__version__}, cuda {torch.version.cuda}, cuda disponible: {torch.cuda.is_available()}")
    from labor2vec.modelo.config import HPARAMS_PAPER, hparams
    from labor2vec.modelo.pretrain import TransformerEncoder

    dispositivo = torch.device(args.dispositivo)
    if dispositivo.type == "cuda":
        paso(f"GPU {torch.cuda.get_device_name(dispositivo)}, capacidad {torch.cuda.get_device_capability(dispositivo)}")
        a = torch.randn(4096, 4096, device=dispositivo)
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(20):
            a @ a
        torch.cuda.synchronize()
        paso(f"matmul fp32 4096²: {20 * 2 * 4096**3 / (time.time() - t) / 1e12:.1f} TFLOP/s")
        del a

    max_length = args.max_length or (HPARAMS_PAPER["max_length"] if dispositivo.type == "cuda" else 512)
    hp = hparams(HPARAMS_PAPER, max_length=max_length)
    torch.manual_seed(0)
    modelo = TransformerEncoder(hp).to(dispositivo)
    n = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    paso(f"modelo del paper en {dispositivo}: {n / 1e6:.2f} M parámetros, max_length {max_length}")

    # redibujar las proyecciones del Performer: se hace al final de cada época del pre-entrenamiento
    modelo.transformer.redraw_projection_matrix(-1)
    paso("proyecciones del Performer redibujadas")

    largo = args.largo or max_length
    for precision in args.precision:
        for lote_n in args.lotes:
            medir(modelo, hp, lote_n, largo, args.pasos, dispositivo, precision)
    paso("fin")


if __name__ == "__main__":
    main()
