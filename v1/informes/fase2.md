# Fase 2 de v1: humo en CPU y piloto de una pasada en Mendieta

Generado el 2026-11-03. Plan: `plans/PLAN_v1.md`. Humo en CPU: ver `informes/fase1.md` (sección «Cambios posteriores»). Piloto: job 1185449 (`l2v_piloto_v1`, partición `short`, nodo ivb10, A30), 09:30 a 10:00 hora de Argentina, `COMPLETED` en 29 min 39 s. Resultados locales en `v1/runs/piloto_v1/` (métricas, log y `modelo_final.pt`, ignorados por git).

## Qué se corrió

`preentrenar.sbatch piloto_v1` con `epocas=1`, `pasos_por_epoca=5375` (una pasada por train: 344.025 / 64), validación cada 1.000 pasos (200 lotes de 32). El resto de la configuración es la de `configs/preentrenamiento.yaml`: el modelo de life2vec con 8,1 millones de parámetros, bf16, lote 32 por 2 de acumulación, vocabulario de 521 tokens y SOP por clase de largo.

## Resultados

| Paso | Perplejidad MLM (unigrama 59,7) | Exactitud MLM | Pérdida SOP | AUC SOP |
|---|---|---|---|---|
| 1.000 | 7,85 | 0,408 | 1,364 | 0,681 |
| 2.000 | 6,34 | 0,438 | 1,311 | 0,753 |
| 3.000 | 5,71 | 0,479 | 1,295 | 0,766 |
| 4.000 | 5,15 | 0,513 | 1,237 | 0,816 |
| 5.000 | 4,96 | 0,524 | 1,209 | 0,833 |

- **Pérdida de entrenamiento:** de 3,18 (primeros 50 pasos) a 1,55 (últimos 50), el 49 %. Máximo 3,98; sin explosiones.
- **SOP:** el AUC sube de forma sostenida hasta 0,833 en una sola pasada. En el humo chico con el SOP arreglado quedaba en 0,52 y en el control sin grilla en 0,56. Con el atajo (SOP sin arreglar) ya daba 0,87 en 2 épocas del humo chico; acá sube de a poco, que es lo esperable cuando se aprende de verdad. Referencia de v0: 0,865 tras 150 épocas. Las recalls por clase todavía oscilan (0,43 a 0,95): falta entrenamiento.
- **Velocidad:** 210 secuencias por segundo de media (mediana 209) y 135 mil tokens por segundo; memoria pico 18,0 GiB de 24. v0 daba 242 secuencias y 115 mil tokens por segundo: v1 mueve más tokens por segundo y procesa menos secuencias porque son más largas (1,4 veces de media). La corrida entera, con 5 validaciones, tardó 1.712 s.
- La última validación es la del paso 5.000; el modelo final no tiene una validación propia al terminar.

## Reparto de la pérdida del MLM por categoría (modelo final, 80 lotes de validación, 290.736 objetivos)

| Categoría | % de objetivos | NLL | Acierto | % de la pérdida |
|---|---|---|---|---|
| salario | 15,2 % | 4,26 | 0,103 | 40,4 % |
| rama (r34) | 15,3 % | 1,68 | 0,432 | 16,0 % |
| tamaño | 15,3 % | 1,43 | 0,472 | 13,5 % |
| inicio de la empresa | 15,3 % | 1,39 | 0,488 | 13,2 % |
| provincia | 15,3 % | 1,38 | 0,497 | 13,1 % |
| `SINREG_SAL` | 16,2 % | 0,088 | 0,977 | 0,9 % |
| `SINREG_INT` | 6,8 % | 0,50 | 0,868 | 2,1 % |
| nacimiento, sexo, rama (r32) | 0,6 % | | | 0,8 % |

- Los `SINREG` son el 23,0 % de los objetivos del MLM y el 3,0 % de la pérdida: casi triviales, y `INT` (NLL 0,50) cuesta más que `SAL` (0,088), como era de esperar. La pérdida media es 1,61 con ellos y 2,03 sin ellos (perplejidad 5,0 y 7,6), así que la perplejidad de v1 no es comparable con la de v0.
- Quedan en el MLM como están (enmascarables como cualquier token): dan señal gratis pero no distorsionan el aprendizaje de lo demás, que sigue siendo el 97 % de la pérdida. Si en el pre-entrenamiento largo la diferencia con v0 fuera chica, se puede revisar bajándoles la probabilidad de enmascarado.
- El salario es lo más difícil de predecir (acierto 10 %), igual que en v0.

## Costo del pre-entrenamiento largo (fase 3)

150 épocas de 375 pasos de lote efectivo 64 son 3,6 millones de secuencias. A unas 200 secuencias por segundo (con validación) son **unas 5 a 6 horas**, bastante menos que las 7 a 10 que estimaba el plan. La partición `short` corta a 1 h, así que son 6 o 7 tramos encadenados con `--dependency=afterany` (`preentrenar.sbatch` se reanuda con el mismo comando).

## Criterios de aceptación de la fase 2

1. Pérdida baja: ✔ (49 % de la inicial).
2. SOP: ✔ el AUC sube hasta 0,833, no es el atajo (el control daba 0,56; la referencia 1,636 de pérdida no se usa, ver el plan).
3. Reparto de la pérdida del MLM entre `SINREG` y el resto: ✔ (tabla de arriba).
4. Velocidad medida en la A30: ✔ (210 seq/s, 18 GiB).
