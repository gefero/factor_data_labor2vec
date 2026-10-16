"""Corpus de eventos: una fila por trabajador, con listas por evento en orden cronológico.

Los códigos son crudos (la conversión a tokens se hace al cargar) y el dato faltante se guarda como -1.
Se procesa el panel por tramos de `id_trabajador`, cada uno con DuckDB acotado, y se escribe un archivo
por tramo y split: `corpus/eventos_{split}/parte_{k}.parquet`.

    python -m labor2vec.tokens.corpus [--muestra 0.01] [--tramos 20]
"""

import argparse
import shutil
import time

from labor2vec.utiles.duck import conectar
from labor2vec.utiles.rutas import CORPUS, PANEL, RELACIONES, SPLITS, TRABAJADORES

SPLITS_NOMBRES = ("train", "val", "test")

# Orden de los eventos: mes, inicio de la relación y, al final, id_relacion (que no es cronológico, ver fase 1)
_SQL_TRAMO = """
with ev as (
    select p.id_trabajador, p.id_relacion, p.mes_idx,
           coalesce(p.pct_rem_mes, -1) as pct_rem,
           coalesce(p.tramo_empleo, -1) as tramo,
           coalesce(p.r34, -1) as r34,
           coalesce(p.provi, -1) as provi,
           coalesce(p.tr_anio_ini, -1) as antig,
           coalesce(r.inicio_mes_idx, p.mes_idx) as inicio,
           coalesce(p.mes_idx - lag(p.mes_idx) over (partition by p.id_trabajador, p.id_relacion
                                                       order by p.mes_idx) > 1, true) as nueva
    from read_parquet('{panel}') p
    join read_parquet('{splits}') s using (id_trabajador)
    left join read_parquet('{relaciones}') r using (id_trabajador, id_relacion)
    where s.split = '{split}' and p.id_trabajador between {desde} and {hasta}{muestra}
)
select e.id_trabajador, any_value(t.sexo)::TINYINT as sexo, any_value(t.fnacim)::SMALLINT as fnacim,
       list(mes_idx order by mes_idx, inicio, id_relacion)::SMALLINT[] as mes_idx,
       list(pct_rem order by mes_idx, inicio, id_relacion)::TINYINT[] as pct_rem,
       list(tramo order by mes_idx, inicio, id_relacion)::TINYINT[] as tramo,
       list(r34 order by mes_idx, inicio, id_relacion)::SMALLINT[] as r34,
       list(provi order by mes_idx, inicio, id_relacion)::TINYINT[] as provi,
       list(antig order by mes_idx, inicio, id_relacion)::TINYINT[] as antig,
       list(id_relacion order by mes_idx, inicio, id_relacion)::SMALLINT[] as id_relacion,
       list(nueva order by mes_idx, inicio, id_relacion) as nueva
from ev e join read_parquet('{trabajadores}') t using (id_trabajador)
group by e.id_trabajador
order by e.id_trabajador
"""


def construir(salida=CORPUS, tramos: int = 20, muestra: float | None = None, panel=PANEL, splits=SPLITS,
              relaciones=RELACIONES, trabajadores=TRABAJADORES) -> dict:
    """Escribe el corpus y devuelve {split: (trabajadores, eventos)}. `muestra` es la fracción de trabajadores
    (elegidos por hash del id, de modo reproducible)."""
    con = conectar()
    ids = con.execute(f"select min(id_trabajador), max(id_trabajador) from read_parquet('{splits}')").fetchone()
    # tramos de igual cantidad de trabajadores: cortes en los cuantiles del id
    cuantiles = ", ".join(f"quantile_disc(id_trabajador, {k / tramos})" for k in range(1, tramos))
    cortes = list(con.execute(f"select {cuantiles} from read_parquet('{splits}')").fetchone()) if tramos > 1 else []
    limites = [ids[0]] + [c + 1 for c in cortes] + [ids[1] + 1]
    filtro = f" and hash(p.id_trabajador) % 1000000 < {int(muestra * 1_000_000)}" if muestra else ""

    for s in SPLITS_NOMBRES:
        shutil.rmtree(salida / f"eventos_{s}", ignore_errors=True)
        (salida / f"eventos_{s}").mkdir(parents=True)
    totales = {s: [0, 0] for s in SPLITS_NOMBRES}
    t0 = time.time()
    for k in range(tramos):
        desde, hasta = limites[k], limites[k + 1] - 1
        for s in SPLITS_NOMBRES:
            destino = salida / f"eventos_{s}" / f"parte_{k:02d}.parquet"
            sql = _SQL_TRAMO.format(panel=panel, splits=splits, relaciones=relaciones, trabajadores=trabajadores,
                                    split=s, desde=desde, hasta=hasta, muestra=filtro)
            con.execute(f"copy ({sql}) to '{destino}' (format parquet, compression zstd)")
            n_t, n_e = con.execute(
                f"select count(*), coalesce(sum(len(mes_idx)), 0) from read_parquet('{destino}')").fetchone()
            totales[s][0] += n_t
            totales[s][1] += n_e
        print(f"tramo {k + 1}/{tramos} ({time.time() - t0:.0f} s): "
              + ", ".join(f"{s} {v[0]:,} trab." for s, v in totales.items()), flush=True)
    return {s: tuple(v) for s, v in totales.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--muestra", type=float, help="fracción de trabajadores (corpus_muestra/ en vez de corpus/)")
    ap.add_argument("--tramos", type=int, default=20)
    args = ap.parse_args()
    salida = CORPUS.with_name("corpus_muestra") if args.muestra else CORPUS
    print(construir(salida, args.tramos, args.muestra))


if __name__ == "__main__":
    main()
