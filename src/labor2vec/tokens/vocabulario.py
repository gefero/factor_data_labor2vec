"""Vocabulario de la MLER, ajustado sólo con train.

Orden de los especiales idéntico al de life2vec-light, para no tocar los `ignore_index = [0,4,5,6,7,8]` del
backbone. Las tablas de búsqueda (`tablas`) convierten códigos crudos del corpus de eventos (con -1 = dato
faltante) en ids de token; lo faltante pasa a [UNK].

    python -m labor2vec.tokens.vocabulario
"""

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from labor2vec.utiles.duck import conectar
from labor2vec.utiles.rutas import PANEL, RAMAS, VOCABULARIO

ESPECIALES = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[PLCH0]", "[PLCH1]", "[PLCH2]", "[PLCH3]", "[PLCH4]", "[UNK]"]
PAD, CLS, SEP, MASK, UNK = 0, 1, 2, 3, 9
N_ESPECIALES = len(ESPECIALES)

UMBRAL_RAMA = 50       # trabajadores en train; por debajo, la rama pasa a su r32
UMBRAL_NACIMIENTO = 100  # trabajadores en train por año de nacimiento; los extremos poco frecuentes se agrupan

SEXOS = {1: "SEXO_M", 2: "SEXO_V"}  # descriptor de la MLER: 1 = Mujer, 2 = Varón


@dataclass
class Vocabulario:
    """Tokens (id = posición en `tokens`) y mapeos de códigos crudos a tokens."""

    tokens: list[str]
    categorias: list[str]
    frecuencias: list[int]
    ramas: pl.DataFrame  # r34, r32, n_trab_train, token
    nac_min: int = 0
    nac_max: int = 0
    _id: dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._id = {t: i for i, t in enumerate(self.tokens)}

    def __len__(self) -> int:
        return len(self.tokens)

    def id(self, token: str) -> int:
        return self._id.get(token, UNK)

    # ---- tablas de búsqueda ------------------------------------------------------------------------
    def _lut(self, codigos: range | list[int], nombre) -> np.ndarray:
        """Tabla indexada por `codigo + 1` (el índice 0 es el dato faltante, -1)."""
        cods = list(codigos)
        lut = np.full(max(cods) + 2, UNK, dtype=np.int32)
        for c in cods:
            lut[c + 1] = self.id(nombre(c))
        return lut

    def tablas(self) -> dict[str, np.ndarray]:
        """LUT por variable. Se indexan con `codigo + 1`; ver `buscar`."""
        r = self.ramas
        rama = np.full(int(r["r34"].max()) + 2, UNK, dtype=np.int32)
        for r34, r32, n, tok in r.iter_rows():
            rama[r34 + 1] = self.id(tok)
        provs = [int(t[5:]) for t, c in zip(self.tokens, self.categorias) if c == "provincia"]
        return {
            "rem": self._lut(range(0, 101), lambda c: f"REM_{c:03d}"),
            "tam": self._lut(range(1, 5), lambda c: f"TAM_{c}"),
            "rama": rama,
            "prov": self._lut(range(0, max(provs) + 1), lambda c: f"PROV_{c:02d}"),
            "ant": self._lut(range(1, 5), lambda c: f"ANT_{c}"),
        }

    def fondo(self, sexo: int, fnacim: int) -> tuple[int, int]:
        """Ids de los dos tokens de background (sexo, año de nacimiento con los extremos agrupados)."""
        nac = min(max(int(fnacim), self.nac_min), self.nac_max)
        return self.id(SEXOS.get(int(sexo), "")), self.id(f"NAC_{nac}")

    # ---- persistencia ------------------------------------------------------------------------------
    def guardar(self, ruta=VOCABULARIO, ruta_ramas=RAMAS) -> None:
        ruta.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(
            {"token": self.tokens, "id": range(len(self.tokens)), "categoria": self.categorias,
             "frecuencia": self.frecuencias},
            schema_overrides={"id": pl.Int32, "frecuencia": pl.Int64},
        ).write_parquet(ruta)
        self.ramas.write_parquet(ruta_ramas)

    @classmethod
    def cargar(cls, ruta=VOCABULARIO, ruta_ramas=RAMAS) -> "Vocabulario":
        v = pl.read_parquet(ruta).sort("id")
        nac = [int(t[4:]) for t, c in zip(v["token"], v["categoria"]) if c == "nacimiento"]
        return cls(v["token"].to_list(), v["categoria"].to_list(), v["frecuencia"].to_list(),
                   pl.read_parquet(ruta_ramas), min(nac), max(nac))

    # ---- ajuste ------------------------------------------------------------------------------------
    @classmethod
    def ajustar(cls, panel=PANEL, umbral_rama: int = UMBRAL_RAMA, umbral_nac: int = UMBRAL_NACIMIENTO,
                con=None) -> "Vocabulario":
        """Ajusta el vocabulario con los eventos de train (todo el período, sin ventana).

        `panel` necesita las columnas del panel de la fase 1 (id_trabajador, pct_rem_mes, tramo_empleo, r32, r34,
        provi, tr_anio_ini, fnacim, sexo, split). El resultado es determinista.
        """
        con = con or conectar()
        base = f"(select * from read_parquet('{panel}') where split = 'train')"
        # frecuencia de cada código crudo = cantidad de eventos de train
        cuentas = {}
        for nombre, col in [("rem", "pct_rem_mes"), ("tam", "tramo_empleo"), ("prov", "provi"),
                            ("ant", "tr_anio_ini")]:
            cuentas[nombre] = dict(con.execute(
                f"select {col}, count(*) from {base} where {col} is not null group by 1").fetchall())
        rama = con.execute(
            f"select r34, any_value(r32), count(distinct id_trabajador), count(*) from {base} "
            f"where r34 is not null group by 1 order by 1").fetchall()
        # r34 -> r32 con todo el panel, para ramas que sólo aparecen en val/test
        r34_r32 = dict(con.execute(
            f"select r34, any_value(r32) from read_parquet('{panel}') where r34 is not null and r32 is not null "
            f"group by 1").fetchall())
        # trabajadores por año de nacimiento y sexo
        nac = dict(con.execute(
            f"select fnacim, count(distinct id_trabajador) from {base} where fnacim is not null group by 1"
        ).fetchall())
        sexo = dict(con.execute(
            f"select sexo, count(distinct id_trabajador) from {base} where sexo is not null group by 1").fetchall())

        tokens, cats, frec = list(ESPECIALES), ["especial"] * N_ESPECIALES, [0] * N_ESPECIALES

        def agregar(tok, cat, f):
            tokens.append(tok), cats.append(cat), frec.append(int(f))

        for cod, nombre in sorted(SEXOS.items(), key=lambda kv: kv[1]):
            agregar(nombre, "sexo", sexo.get(cod, 0))
        # nacimiento: los años de los extremos con pocos trabajadores se agrupan en el primero/último frecuente
        frecuentes = sorted(a for a, n in nac.items() if n >= umbral_nac)
        lo, hi = frecuentes[0], frecuentes[-1]
        for a in range(lo, hi + 1):
            f = sum(n for b, n in nac.items() if (b == a) or (a == lo and b < lo) or (a == hi and b > hi))
            agregar(f"NAC_{a}", "nacimiento", f)
        for c in range(0, 101):
            agregar(f"REM_{c:03d}", "salario", cuentas["rem"].get(c, 0))
        for c in range(1, 5):
            agregar(f"TAM_{c}", "tamaño", cuentas["tam"].get(c, 0))
        # ramas: r34 con suficientes trabajadores; el resto, a su r32
        tabla_ramas = []
        r32_necesarias = {}
        for r34, _, n_trab, n_ev in rama:
            if n_trab >= umbral_rama:
                agregar(f"RAMA_{r34}", "rama", n_ev)
        r34_en_train = {r[0]: r for r in rama}
        for r34 in sorted(r34_r32):
            r32 = r34_r32[r34]
            n_trab = r34_en_train[r34][2] if r34 in r34_en_train else 0
            if n_trab >= umbral_rama:
                tabla_ramas.append((r34, r32, n_trab, f"RAMA_{r34}"))
            else:
                r32_necesarias[r32] = r32_necesarias.get(r32, 0) + (r34_en_train[r34][3] if r34 in r34_en_train else 0)
                tabla_ramas.append((r34, r32, n_trab, f"RAMA2_{r32}"))
        for r32 in sorted(r32_necesarias):
            agregar(f"RAMA2_{r32}", "rama2", r32_necesarias[r32])
        for c in sorted(cuentas["prov"]):
            agregar(f"PROV_{c:02d}", "provincia", cuentas["prov"][c])
        for c in range(1, 5):
            agregar(f"ANT_{c}", "antigüedad", cuentas["ant"].get(c, 0))

        ramas = pl.DataFrame(tabla_ramas, schema=["r34", "r32", "n_trab_train", "token"], orient="row",
                             schema_overrides={"r34": pl.Int32, "r32": pl.Int32, "n_trab_train": pl.Int64})
        return cls(tokens, cats, frec, ramas, lo, hi)


def main() -> None:
    v = Vocabulario.ajustar()
    v.guardar()
    resumen = {c: v.categorias.count(c) for c in dict.fromkeys(v.categorias)}
    print(f"{len(v)} tokens -> {VOCABULARIO}\n{resumen}")


if __name__ == "__main__":
    main()
