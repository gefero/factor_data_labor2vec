"""Validación del panel contra los cuadros "Muestra" del Anexo estadístico del MTEySS.

Recalcula desde panel_base (todos los trabajadores, sin exclusiones) las cifras publicadas de la
muestra y las compara: empleo mensual (C1.1), por sexo (C1.2), por letra (C1.3), por provincia
(C1.5), por tamaño (C1.6), por rama a 4 dígitos (C1.8), remuneración media mensual (C2.1) y
percentiles (C2.2). "Empleo" = cantidad de filas (relación × mes), incluidas las remuneraciones 0.

Salida: informes/validacion_mler.md

    python -m labor2vec.datos.validacion
"""

import unicodedata
from datetime import datetime

import openpyxl
import polars as pl

from labor2vec.utiles.duck import conectar
from labor2vec.utiles.rutas import DESCRIPTORES, INFORMES, PANEL_BASE, XLSX_ANEXO

DICIEMBRES = [199612, 200512, 201512, 202112]
COLS_MUESTRA_DIC = [6, 7, 8, 9]  # columnas "Muestra" de los cuadros con cortes en diciembre


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return " ".join(s.lower().split())


def _aaaamm(v) -> int | None:
    if isinstance(v, datetime):
        return v.year * 100 + v.month
    return None


def _filas(wb, hoja: str) -> list[tuple]:
    return list(wb[hoja].iter_rows(values_only=True))


def _comparar(nombre: str, oficial: pl.DataFrame, propio: pl.DataFrame, claves: list[str],
              tol_rel: float = 0.0) -> dict:
    """Une por `claves` y compara las columnas `oficial` y `propio`."""
    df = oficial.join(propio, on=claves, how="left").with_columns(
        pl.col("propio").fill_null(0),
    ).with_columns(
        (pl.col("propio") - pl.col("oficial")).alias("dif"),
    ).with_columns(
        (pl.col("dif").abs() / pl.col("oficial").abs().clip(lower_bound=1e-9)).alias("dif_rel"),
    )
    ok = df.filter(pl.col("dif_rel") <= tol_rel) if tol_rel else df.filter(pl.col("dif") == 0)
    return {
        "cuadro": nombre,
        "comparaciones": df.height,
        "coinciden": ok.height,
        "max_dif_abs": float(df["dif"].abs().max()),
        "max_dif_rel": float(df["dif_rel"].max()),
        "tolerancia": "exacta" if not tol_rel else f"{tol_rel:.1%}",
        "peores": df.sort(pl.col("dif").abs(), descending=True).head(5),
    }


def validar() -> list[dict]:
    wb = openpyxl.load_workbook(XLSX_ANEXO, read_only=True, data_only=True)
    con = conectar()
    P = f"'{PANEL_BASE}'"
    res = []

    # C1.1 empleo mensual
    of = pl.DataFrame([{"tiempo": _aaaamm(r[1]), "oficial": r[3]} for r in _filas(wb, "C 1.1")
                       if _aaaamm(r[1]) and r[3] is not None])
    pr = con.execute(f"SELECT tiempo, count(*) AS propio FROM {P} GROUP BY tiempo").pl()
    res.append(_comparar("C1.1 empleo mensual", of, pr, ["tiempo"]))

    # C1.2 por sexo (Mujer, Hombre, sin dato)
    filas = []
    for r in _filas(wb, "C 1.2"):
        t = _aaaamm(r[1])
        if t and r[6] is not None:
            filas += [{"tiempo": t, "sexo": s, "oficial": r[c]} for s, c in ((1, 6), (2, 7), (0, 8))]
    of = pl.DataFrame(filas)
    pr = con.execute(f"SELECT tiempo, coalesce(sexo, 0) AS sexo, count(*) AS propio FROM {P} GROUP BY ALL").pl()
    res.append(_comparar("C1.2 empleo mensual por sexo", of, pr.cast({"sexo": pl.Int64}), ["tiempo", "sexo"]))

    # C1.3 por letra: bloque "Empleo muestra"
    letras = pl.read_parquet(DESCRIPTORES / "letra.parquet").drop_nulls("letra_ciiu")
    cod_letra = dict(zip(letras["letra_ciiu"], letras["letra"]))
    filas3 = _filas(wb, "C 1.3")
    ini = next(i for i, r in enumerate(filas3) if r[1] == "Empleo muestra")
    encabezado = filas3[ini + 1]
    filas = []
    for r in filas3[ini + 2:]:
        if r[1] == "Total" or r[1] is None:
            break
        for j, h in enumerate(encabezado):
            if _aaaamm(h) and r[j] is not None:
                filas.append({"tiempo": _aaaamm(h), "letra": cod_letra[r[1]], "oficial": r[j]})
    of = pl.DataFrame(filas)
    pr = con.execute(f"SELECT tiempo, letra, count(*) AS propio FROM {P} WHERE letra IS NOT NULL GROUP BY ALL").pl()
    res.append(_comparar("C1.3 empleo mensual por letra", of, pr.cast({"letra": pl.Int64}), ["tiempo", "letra"]))

    # C1.5 por provincia (diciembres)
    provs = pl.read_parquet(DESCRIPTORES / "provi.parquet")
    cod_prov = {_norm(d): c for c, d in zip(provs["provi"], provs["descripcion"])}
    cod_prov |= {"caba": 2, "resto prov. bs as": 6, "partidos de gba": 1, "sin asignar": 0}
    res.append(_por_diciembre(wb, con, "C 1.5", "C1.5 empleo por provincia (dic.)", "provi",
                              lambda s: cod_prov.get(_norm(s))))

    # C1.6 por tamaño (diciembres)
    cod_tramo = {"hasta 10": 4, "mas de 10 y hasta 50 empleados": 3,
                 "mas de 50 y hasta 200 empleados": 2, "mas de 200 empleados": 1, "sin dato": 0}
    res.append(_por_diciembre(wb, con, "C 1.6", "C1.6 empleo por tamaño (dic.)", "tramo_empleo",
                              lambda s: cod_tramo.get(_norm(s))))

    # C1.8 por rama a 4 dígitos (diciembres)
    res.append(_por_diciembre(wb, con, "C 1.8", "C1.8 empleo por rama r34 (dic.)", "r34",
                              lambda s: s if isinstance(s, int) else (0 if _norm(s) == "sin dato" else None)))

    # C2.1 remuneración media mensual: se prueban varias definiciones
    of = pl.DataFrame([{"tiempo": _aaaamm(r[1]), "oficial": r[3]} for r in _filas(wb, "C 2.1")
                       if _aaaamm(r[1]) and isinstance(r[3], (int, float))])
    definiciones = {
        "todas las filas": "true",
        "rem_tot > 0": "rem_tot > 0",
        "rem_tot >= 30": "rem_tot >= 30",
    }
    for etiqueta, cond in definiciones.items():
        pr = con.execute(f"SELECT tiempo, avg(rem_tot) FILTER (WHERE {cond}) AS propio FROM {P} GROUP BY tiempo").pl()
        res.append(_comparar(f"C2.1 remuneración media ({etiqueta})", of, pr, ["tiempo"], tol_rel=0.005))

    # C2.2 percentiles (diciembres)
    filas = []
    for r in _filas(wb, "C 2.2"):
        nombre = r[1]
        if isinstance(nombre, str) and (nombre.startswith("Percentil") or nombre == "Mediana"):
            p = 50 if nombre == "Mediana" else int(nombre.split()[-1])
            filas += [{"tiempo": t, "p": p, "oficial": r[c]} for t, c in zip(DICIEMBRES, COLS_MUESTRA_DIC)]
    of = pl.DataFrame(filas)
    ps = sorted(of["p"].unique().to_list())
    for etiqueta, func, cond in (("cont, rem_tot > 0", "quantile_cont", "rem_tot > 0"),
                                 ("disc, todas", "quantile_disc", "true")):
        pr = con.execute(f"""
            SELECT tiempo, unnest({ps}) AS p, unnest({func}(rem_tot, {[p / 100 for p in ps]})) AS propio
            FROM {P} WHERE tiempo IN {tuple(DICIEMBRES)} AND {cond} GROUP BY tiempo""").pl()
        res.append(_comparar(f"C2.2 percentiles (dic.; {etiqueta})", of, pr.cast({"p": pl.Int64}),
                             ["tiempo", "p"], tol_rel=0.005))
    return res


def _por_diciembre(wb, con, hoja, nombre, variable, codificar) -> dict:
    filas = []
    for r in _filas(wb, hoja):
        cod = codificar(r[1]) if r[1] is not None else None
        if cod is None or not isinstance(r[6], (int, float)):
            continue
        filas += [{"tiempo": t, variable: cod, "oficial": r[c]}
                  for t, c in zip(DICIEMBRES, COLS_MUESTRA_DIC) if r[c] is not None]
    of = pl.DataFrame(filas)
    pr = con.execute(f"""
        SELECT tiempo, coalesce({variable}, 0) AS {variable}, count(*) AS propio
        FROM '{PANEL_BASE}' WHERE tiempo IN {tuple(DICIEMBRES)} GROUP BY ALL""").pl()
    return _comparar(nombre, of, pr.cast({variable: pl.Int64}), ["tiempo", variable])


def informe(res: list[dict]) -> str:
    lineas = [
        "# Validación del panel contra los cuadros oficiales de la MLER",
        "",
        f"Generado por `python -m labor2vec.datos.validacion` ({datetime.now():%Y-%m-%d %H:%M}).",
        "Fuente oficial: columnas \"Muestra\" del *Anexo estadístico y descriptores* (MTEySS).",
        "Empleo = filas (relación × mes), incluidas las remuneraciones 0. Remuneraciones: tolerancia 0,5 %.",
        "",
        "| Cuadro | Comparaciones | Coinciden | Tolerancia | Máx. dif. absoluta | Máx. dif. relativa |",
        "|---|---|---|---|---|---|",
    ]
    for r in res:
        lineas.append(f"| {r['cuadro']} | {r['comparaciones']:,} | {r['coinciden']:,} "
                      f"({r['coinciden'] / r['comparaciones']:.1%}) | {r['tolerancia']} | "
                      f"{r['max_dif_abs']:,.2f} | {r['max_dif_rel']:.3%} |")
    lineas += ["", "## Mayores diferencias por cuadro", ""]
    for r in res:
        if r["coinciden"] == r["comparaciones"]:
            continue
        lineas += [f"### {r['cuadro']}", "", "```", str(r["peores"]), "```", ""]
    return "\n".join(lineas)


if __name__ == "__main__":
    pl.Config.set_tbl_rows(10)
    res = validar()
    INFORMES.mkdir(exist_ok=True)
    salida = INFORMES / "validacion_mler.md"
    salida.write_text(informe(res))
    for r in res:
        print(f"{r['cuadro']:<45} {r['coinciden']:>6}/{r['comparaciones']:<6} "
              f"máx. dif. rel. {r['max_dif_rel']:.3%}")
    print(f"informe en {salida}")
