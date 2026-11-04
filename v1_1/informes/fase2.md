# v1.1: humo en CPU y piloto en Mendieta

Generado el 2026-11-04. Plan: `plans/PLAN_v1_1.md` (pasos 3 y 4). Código: `v1_1/` (copia de v1 con las opciones `sop_sinreg_fijo`, `mlm_sinreg: entrada` y `mask_base: sin_sinreg`; 198 tests pasan y 6 se saltan, igual que en v1).

## Humo en CPU (`runs/humo_cpu_v11`)

Modelo chico, muestra del 1 %, 15 épocas, 8 min (2026-11-03, 23:17 a 23:25 hora de Argentina). Entrenó sin errores; pérdida de entrenamiento del MLM de 4,85 a 3,51.

| | Perplejidad final | Unigrama | AUC SOP |
|---|---|---|---|
| Control sin grilla (= v0), humo de v1 | 32,0 | 78,7 | 0,56 |
| v1 (con el SOP por clase de largo) | 19,1 | 61,0 | 0,52 |
| **v1.1** | **31,4** | **78,6** | **0,56** |

Sin los `SINREG` como objetivos, la perplejidad y el unigrama vuelven a la escala de v0. Las métricas por grupo (`val/mlm_*_empleo` y `_fondo`) se registran, y la de `SINREG` no aparece porque ya no hay objetivos de ese tipo.

## Piloto en Mendieta (`runs/piloto_v11`, job 1185608)

Una pasada por train (5.375 pasos de lote 64), el mismo comando que el piloto de v1. Corrió del 2026-11-03 a las 23:28 al 23:58, hora de Argentina (29 min 57 s): COMPLETED, con `modelo_final.pt` guardado y sin NaN. La pérdida de entrenamiento del MLM bajó de 4,69 a 2,01.

| | v1 (piloto) | v1.1 (piloto) |
|---|---|---|
| Secuencias por segundo | 210 | ~227 |
| Memoria pico de la GPU | 18,0 GiB | 18,0 GiB |
| Perplejidad de validación (paso 5.000) | 4,96 (unigrama 59,7) | 6,54 (unigrama 82,1) |
| AUC del SOP (paso 5.000) | 0,833 | 0,773 |

Validación de v1.1 a lo largo de la pasada:

| Paso | Perplejidad | NLL de empleo | Acierto de empleo | NLL de fondo | AUC SOP | Recalls SOP (normal / invertido / mezclado) |
|---|---|---|---|---|---|---|
| 1.000 | 11,91 | 2,474 | 0,264 | 3,00 | 0,559 | 0,99 / 0,01 / 0,00 |
| 2.000 | 8,93 | 2,188 | 0,324 | 2,25 | 0,654 | 0,86 / 0,23 / 0,11 |
| 3.000 | 7,92 | 2,070 | 0,373 | 2,02 | 0,687 | 0,92 / 0,23 / 0,11 |
| 4.000 | 6,72 | 1,905 | 0,425 | 1,91 | 0,751 | 0,83 / 0,37 / 0,42 |
| 5.000 | 6,54 | 1,878 | 0,429 | 1,83 | 0,773 | 0,89 / 0,33 / 0,41 |

Lectura:

- **Funciona igual que v1 en la GPU.** Es algo más rápido (en v1.1 hay menos objetivos del MLM por documento) y usa la misma memoria.
- **La perplejidad y la AUC del SOP dan más que en v1, y es lo esperado.** v1 tenía un 23 % de objetivos `SINREG` casi gratis y un atajo en el SOP. Las cifras de v1.1 no se comparan con las de v1 sino con las de v0. El unigrama de 82,1 es prácticamente el de v0 (81,4), porque ahora los objetivos son los mismos casos que en v0.
- **El SOP aprende sin atajo:** la AUC sube de 0,56 a 0,77 en una pasada.
- **Estimación para 150 épocas:** a ~227 secuencias por segundo, unos 11.000 pasos por tramo de 55 min en un nodo rápido, y unos 8.000 en uno lento (como `ivb06` en v1). Los 56.250 pasos son entonces entre 5 y 7 tramos `short`, unas 5 a 6 h.

Siguiente: pre-entrenamiento `pre_150ep` de v1.1 (paso 5 del plan), con OK para encolar.
