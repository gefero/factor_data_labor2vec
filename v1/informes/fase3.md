# Fase 3 de v1: pre-entrenamiento de 150 épocas (`pre_150ep`, semilla 0)

Generado el 2026-11-03. Plan: `plans/PLAN_v1.md`. Corrida: 7 jobs de Mendieta encadenados con `afterok` (1185456 a 1185461 y el de seguro 1185517), de las 10:41 a las 16:15 hora de Argentina. Resultados locales en `v1/runs/pre_150ep/` (config, métricas de 7 versiones del log, logs de los jobs y `modelo_final.pt`; ignorados por git). Mismo calendario y semilla que el `pre_150ep` de v0: 150 épocas de 375 pasos de lote efectivo 64 (56.250 pasos, 3,6 millones de secuencias), con la codificación de grilla y el SOP por clase de largo.

## Cómo corrió

- Tramos 1 a 4 en `ivb12` (55 min cada uno, unos 10.400 pasos, 209 secuencias por segundo). Tramos 5 y 6 en `ivb06`, más lento (185 secuencias por segundo, unos 8.000 pasos por hora). El tramo 6 terminó el entrenamiento en 48 min 39 s.
- El tramo 7 (de seguro, por la lentitud de `ivb06`) arrancó apenas terminó el 6 y no hizo nada: restauró `last.ckpt`, vio que `max_epochs=150` estaba cumplido y salió en 35 s. El modelo es el del paso 56.250.
- Memoria pico 18,0 GiB de 24. Sin NaN ni errores. Pérdida de entrenamiento de 3,19 (primeros 50 pasos) a 1,41 (últimos 50).

## Validación (200 lotes de 32, igual que en v0)

| Paso | Perplejidad MLM (unigrama 59,7) | Exactitud MLM | Pérdida SOP | AUC SOP |
|---|---|---|---|---|
| 1.125 | 11,52 | 0,392 | 1,377 | 0,620 |
| 6.750 | 4,82 | 0,523 | 1,244 | 0,799 |
| 12.375 | 4,64 | 0,530 | 1,195 | 0,831 |
| 23.625 | 4,47 | 0,533 | 1,163 | 0,867 |
| 34.875 | 4,12 | 0,546 | 1,134 | 0,884 |
| 46.125 | 3,99 | 0,551 | 1,104 | 0,904 |
| 56.250 | **3,95** | **0,553** | **1,091** | **0,913** |

La validación mejora hasta el final y se aplana en los últimos 5.000 pasos (calendario de tasa de aprendizaje decreciente); sin señales de sobreajuste (51 validaciones).

## Comparación con v0 (`pre_150ep`)

| | v0 | v1 |
|---|---|---|
| AUC del SOP | 0,865 | **0,913** |
| Pérdida del SOP (mismos pesos de clase) | 1,196 | **1,091** |
| Recall del SOP (normal / invertido / mezclado) | 0,91 / 0,39 / 0,63 | **0,95 / 0,57 / 0,67** |
| Perplejidad del MLM | 4,73 | 3,95 *no comparable: el 23 % de los objetivos de v1 son `SINREG`, casi gratis* |

> **Corrección (2026-11-03, noche; `fase3_evaluacion.md`, sección 6):** la mejora del SOP de v1 se debe a un atajo (un `SINREG` fuera de lugar delata el 35 % de los documentos alterados). Sin el atajo, v1 rinde como v0 (AUC pareada 0,782 contra 0,759-0,776 de tres semillas de v0). Además, sobre los mismos casos tapados, la diferencia del MLM es +0,11 nats (no +0,18) y diez veces el ruido de semilla.

El SOP es comparable (mismas clases, mismos pesos) y v1 lo resuelve mejor. Hay que tener presente que en v1 el SOP permuta sólo bloques del mismo largo, así que no es la misma tarea que en v0: puede ser más fácil o más difícil; no es un resultado de v1 «mejor» sin más.

## Pérdida del MLM por categoría, sobre los mismos trabajadores

Los dos modelos finales se evaluaron sobre los **mismos 3.696 trabajadores de validación** (uno de cada 20), con el enmascarado determinista por trabajador. Es la cantidad que sí se puede comparar entre v0 y v1: cuánto cuesta predecir cada variable de un evento de empleo cuando está oculta. NLL = error medio en nats (menor es mejor); el intervalo es el 95 % por remuestreo de trabajadores.

| Variable | NLL v0 | NLL v1 | Diferencia (v1 − v0) |
|---|---|---|---|
| salario | 3,744 | 3,941 | +0,197 [+0,183, +0,211] |
| rama | 1,065 | 1,265 | +0,200 [+0,190, +0,212] |
| tamaño | 1,032 | 1,199 | +0,167 [+0,160, +0,174] |
| provincia | 0,993 | 1,147 | +0,154 [+0,146, +0,161] |
| inicio de la empresa (`ANT` en v0) | 0,955 | 1,141 | +0,187 [+0,180, +0,193] |
| año de nacimiento | 1,384 | 1,840 | +0,456 [+0,403, +0,505] |
| sexo | 0,675 | 0,707 | +0,032 [+0,001, +0,062] |
| **todas, sin `SINREG`** | **1,556** | **1,737** | **+0,181 [+0,175, +0,188]** |

Los `SINREG` de v1 son el 22,7 % de los objetivos del MLM y cuestan 0,143 en promedio (en la evaluación con 80 lotes: `SINREG_SAL` 0,064 con 98 % de acierto, `SINREG_INT` 0,375 con 89 %).

**Lectura.** Con el mismo modelo (8,1 millones de parámetros) y los mismos 56.250 pasos, v1 predice **peor todas las variables de empleo que v0**, unos 0,15 a 0,20 nats cada una, con intervalos que no incluyen el cero. No es un efecto de qué documentos tocaron en la muestra. Lo que no sabemos es por qué. Dos hipótesis, sin probar:

1. **Capacidad y presupuesto.** El modelo es el mismo, pero v1 tiene que aprender además los patrones de `SINREG` y trabajar con secuencias 1,4 veces más largas (media) con el mismo número de pasos. Podría quedar menos capacidad para las variables.
2. **Ventana local de la atención.** Las cabezas locales ven 38 tokens (unos 6 eventos de empleo en v0). En v1 los meses sin registro ocupan posiciones (2 tokens por mes), así que alrededor de un token oculto puede haber menos eventos de empleo a la vista. Importa sobre todo para rama, provincia, empresa y tamaño, que se copian de los eventos vecinos de la misma relación.

Esto **no dice que v1 sea peor para las tareas**: el MLM sobre atributos es una medida indirecta de la calidad de la representación. Lo que decide es el ajuste fino (fase 4, T1 a T4) contra v0 y contra las líneas de base. Pero es una advertencia razonable a mirar ahí, y si T2 a T4 empeoran respecto de v0 se vuelven prioritarios los controles de abajo.

## Controles posibles si hiciera falta (no se corrieron)

- Pre-entrenar v1 sin `SINREG` como objetivos del MLM (se ven como entrada, no se enmascaran): descarta la hipótesis de que dan una señal fácil que distrae.
- Ventana local mayor (por ejemplo 64 en vez de 38 tokens) o modelo más grande: prueba las hipótesis 1 y 2.
- Medir la pérdida por categoría separando si el token oculto tiene o no eventos de empleo vecinos de la misma relación dentro de la ventana.

## Criterios de aceptación de la fase 3

1. Convergencia y perplejidad por categoría: ✔ (tablas de arriba).
2. Sin NaN y con el modelo final guardado (`modelo_final.pt`, 32,6 MB): ✔.
3. Tiempo: 5 h 28 min de reloj para toda la cadena, dentro de la estimación de 5 a 6 h.

Siguiente: fase 4, ajuste fino de T1 a T4 con 3 semillas, GRU sobre los documentos de v1 y comparación con v0 y con las líneas de base de v0 (necesita OK para encolar).
