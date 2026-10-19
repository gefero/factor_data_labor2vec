# Fase 2: tokenización, corpus y carga de datos

Generado el 2026-10-16. Código en `src/labor2vec/tokens/`, pruebas en `tests/test_tokens.py` (18 pruebas nuevas, 31 en total).

## Corpus de eventos

`python -m labor2vec.tokens.corpus` (20 tramos de `id_trabajador`, DuckDB con 4 GB y 4 hilos). Tarda 82 s en el contenedor y ocupa 78 MB.

| Split | Trabajadores | Eventos |
|---|---|---|
| train | 402.949 | 33.859.883 |
| val | 86.347 | 7.230.196 |
| test | 86.346 | 7.240.706 |

La suma de eventos coincide con las filas del panel de los elegibles en los tres splits, y los splits no se solapan.

## Vocabulario (519 tokens, ajustado con train)

| Categoría | Tokens |
|---|---|
| Especiales (orden de `life2vec-light`) | 10 |
| Sexo (1 = Mujer, 2 = Varón en la MLER) | 2 |
| Año de nacimiento (extremos con < 100 trabajadores agrupados) | 77 |
| Salario (`REM_000` a `REM_100`) | 101 |
| Tamaño de empresa | 4 |
| Rama r34 (≥ 50 trabajadores en train) | 282 |
| Rama r32 (reemplaza a las r34 raras) | 14 |
| Provincia | 25 |
| Antigüedad de la empresa | 4 |

## Largos de los documentos (train, tokens antes de la augmentación)

| Variante | Trabajadores | Mediana | p90 | p99 | Truncados a 2.560 |
|---|---|---|---|---|---|
| A, toda la historia | 402.949 | 298 | 1.348 | 1.882 | 0,24 % |
| A, ventana hasta 2015-12 | 344.025 | 268 | 1.156 | 1.456 | 0,12 % |
| B, toda la historia | 402.949 | 163 | 694 | 955 | 0,01 % |
| B, ventana hasta 2015-12 | 344.025 | 148 | 598 | 751 | 0,01 % |

La población de pre-entrenamiento con la ventana a 2015-12 es de **344.025 trabajadores** (el subplan estimaba ~460 k).

## Rendimiento de carga

~3.800 secuencias/s por proceso (variante A con ventana, con augmentación, SOP y MLM). La GPU procesaba ~60 secuencias/s con 2.560 tokens fijos (fase 0), así que la carga no es el cuello de botella; con largo variable el pre-entrenamiento debería salir bastante más barato (mediana de 268 tokens contra 2.560).

## Lote real en el modelo

Con la muestra del 1 %, un lote de 32 documentos (forma (32, 4, 320), objetivos MLM (32, 96)) pasa por el modelo con la configuración del paper (`vocab_size=519`, `n_tokens_fondo=4`) en CPU: pérdida inicial 5,94 (ln 519 ≈ 6,25). La atención local acepta largos múltiplos de 64. Una prueba automática con el modelo chico muestra que la pérdida baja en 40 pasos sobre un lote fijo.

## Decisiones y supuestos

- Ventana del pre-entrenamiento: hasta 2015-12 (`fin_ventana=239`); población: todos los elegibles con eventos en la ventana; variante A como base (la B es la opción `eventos_compactos=True`); umbral de ramas raras: 50 trabajadores.
- El vocabulario se ajusta con todo el período de train, para que sirva igual con o sin ventana.
- **Edad**: `pretrain.py` hace `input_ids.long()`, así que entra como entero (años cumplidos aproximados). La edad fraccionaria exigiría cambiar ese cast.
- **Segmentos**: los eventos recorren el ciclo 2, 3, 1 y el background va en 1 (`CICLO_SEGMENTOS`), igual que `segment_pattern = [2, 3, 1]` del repo completo de life2vec. (La primera versión usaba sólo 2 y 3; corregido el 2026-10-19 tras revisar ese repo.)
- **Quitar tokens** (p = 0,01): borra el 10 % de los tokens; ese porcentaje es un supuesto.
- **SOP**: invertir o mezclar mueve los tokens y deja fijas posición absoluta, edad y segmento. Proporciones 90/5/5 y pesos de clase = inversa de las proporciones (`sop_proporciones`).
- Datos faltantes en el corpus: -1; en el vocabulario pasan a `[UNK]`.
