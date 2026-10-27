# Fase 9, etapa B: intervenciones (generado por `analisis.etapa_b`)

## B3. Intervenciones sobre la historia

Se cambia una cosa de la historia de cada persona del test y se vuelve a correr el modelo. «Afectadas» son las personas cuyo vector cambió (si la intervención no toca nada en su historia, queda igual). El cambio del logit es la predicción de la clase 1 menos la de la clase 0 (sólo en los modelos ajustados). Es lo que el modelo hace con la historia, no un efecto causal en el mundo real.

- `sin_ultimo_anio`: se borran los eventos de los últimos 12 meses.
- `sin_cambios_recientes`: la última relación conserva la rama y el tamaño de la anterior.
- `salario_arriba` / `salario_abajo`: el percentil de los eventos del último año sube o baja 10 puntos.
- `atrasar_2_anios`: todas las fechas, y con ellas las edades, pasan 24 meses atrás.

### t1

| intervención | afectadas | Δ logit medio | mediana | p5 a p95 | % sube | AUL antes → después | Δ salario implícito (pts) | cambia de tipo |
|---|---|---|---|---|---|---|---|---|
| sin_ultimo_anio | 18,337 (100 %) | +0.113 | +0.091 | -0.08 a +0.39 | 87 | 0.613 → 0.610 | -1.78 | – |
| sin_cambios_recientes | 12,792 (70 %) | +0.075 | +0.010 | -0.44 a +0.77 | 54 | 0.613 → 0.605 | -1.49 | – |
| salario_arriba | 18,267 (100 %) | -0.035 | -0.039 | -0.16 a +0.12 | 24 | 0.613 → 0.612 | +2.22 | – |
| salario_abajo | 18,293 (100 %) | +0.080 | +0.068 | -0.00 a +0.21 | 95 | 0.613 → 0.616 | -2.58 | – |
| atrasar_2_anios | 18,337 (100 %) | +0.194 | +0.186 | -0.01 a +0.42 | 94 | 0.613 → 0.611 | -2.03 | – |

### t3

| intervención | afectadas | Δ logit medio | mediana | p5 a p95 | % sube | AUL antes → después | Δ salario implícito (pts) | cambia de tipo |
|---|---|---|---|---|---|---|---|---|
| sin_ultimo_anio | 15,316 (100 %) | +0.222 | +0.212 | -0.36 a +0.89 | 82 | 0.726 → 0.707 | -1.47 | – |
| sin_cambios_recientes | 10,639 (69 %) | -0.030 | -0.100 | -1.86 a +2.40 | 40 | 0.726 → 0.666 | -2.11 | – |
| salario_arriba | 15,259 (100 %) | -0.049 | -0.052 | -0.36 a +0.28 | 34 | 0.726 → 0.726 | +2.87 | – |
| salario_abajo | 15,290 (100 %) | +0.067 | +0.062 | -0.17 a +0.32 | 73 | 0.726 → 0.725 | -2.60 | – |
| atrasar_2_anios | 15,316 (100 %) | +0.211 | +0.203 | -0.38 a +0.83 | 75 | 0.726 → 0.724 | -0.60 | – |

### pre

| intervención | afectadas | Δ logit medio | mediana | p5 a p95 | % sube | AUL antes → después | Δ salario implícito (pts) | cambia de tipo |
|---|---|---|---|---|---|---|---|---|
| sin_ultimo_anio | 18,337 (100 %) | – | – | – | – | – | -1.11 | 11 % (11 % de las afectadas) |
| sin_cambios_recientes | 11,568 (63 %) | – | – | – | – | – | -1.79 | 18 % (28 % de las afectadas) |
| salario_arriba | 18,232 (99 %) | – | – | – | – | – | +1.16 | 6 % (6 % de las afectadas) |
| salario_abajo | 18,279 (100 %) | – | – | – | – | – | -1.20 | 7 % (7 % de las afectadas) |
| atrasar_2_anios | 18,337 (100 %) | – | – | – | – | – | -0.45 | 9 % (9 % de las afectadas) |

«Δ salario implícito»: cambio medio, entre las afectadas, de lo que una sonda lineal (ridge, ajustada en val sobre el propio vector) predice para el percentil salarial de 2015. Si el modelo registra el salario, `salario_arriba` debería dar un valor positivo cercano a +10 y `salario_abajo` uno negativo.

