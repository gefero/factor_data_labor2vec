# Comparaciones de la fase 8 (generado por `tareas.ablaciones`)

Media entre semillas en test; diferencia pareada con IC al 95 % por bootstrap sobre las mismas personas. `*` = el IC no incluye el 0.

### Control «tiempo», sin reajustar

| | entrada normal | control tiempo | diferencia [IC 95 %] |
|---|---|---|---|
| T1 (AUL) | 0.619 | 0.588 | +0.031 [+0.026, +0.035] * |
| T2 (QWK) | 0.830 | 0.765 | +0.065 [+0.059, +0.070] * |
| T3 (AUL) | 0.725 | 0.708 | +0.017 [+0.011, +0.023] * |
| T4 (AUC macro) | 0.772 | 0.701 | +0.071 [+0.064, +0.078] * |

### Control «orden», sin reajustar

| | entrada normal | control orden | diferencia [IC 95 %] |
|---|---|---|---|
| T1 (AUL) | 0.619 | 0.578 | +0.041 [+0.036, +0.045] * |
| T2 (QWK) | 0.830 | 0.804 | +0.026 [+0.022, +0.029] * |
| T3 (AUL) | 0.725 | 0.711 | +0.013 [+0.008, +0.018] * |
| T4 (AUC macro) | 0.772 | 0.739 | +0.034 [+0.029, +0.039] * |

### Reajuste con el tiempo destruido

Se reajustó el transformer pre-entrenado con las fechas y edades permutadas entre los eventos.

| | original | tiempo destruido | diferencia [IC 95 %] |
|---|---|---|---|
| T3 (AUL) | 0.725 | 0.710 | +0.015 [+0.011, +0.019] * |
| T4 (AUC macro) | 0.772 | 0.738 | +0.035 [+0.030, +0.040] * |

### T1: con y sin atributos de último valor (AUL)

| | con todos | sin último valor | diferencia [IC 95 %] |
|---|---|---|---|
| Transformer pre-entrenado | 0.619 | 0.606 | +0.013 [+0.006, +0.019] * |
| Transformer sin pre-entrenar | 0.563 | 0.539 | +0.024 [+0.019, +0.029] * |
| GRU | 0.616 | 0.585 | +0.030 [+0.023, +0.038] * |
| FFNN | 0.608 | 0.583 | +0.025 [+0.020, +0.031] * |
| Regresión logística | 0.617 | 0.585 | +0.032 [+0.025, +0.040] * |

### T2: con y sin atributos de último valor (QWK)

| | con todos | sin último valor | diferencia [IC 95 %] |
|---|---|---|---|
| Transformer pre-entrenado | 0.830 | 0.822 | +0.007 [+0.005, +0.010] * |
| Transformer sin pre-entrenar | 0.745 | 0.818 | -0.073 [-0.080, -0.066] * |
| GRU | 0.828 | 0.822 | +0.006 [+0.003, +0.009] * |
| FFNN | 0.765 | 0.753 | +0.012 [+0.008, +0.015] * |
| Regresión logística | 0.758 | 0.746 | +0.012 [+0.008, +0.017] * |

### T3: con y sin atributos de último valor (AUL)

| | con todos | sin último valor | diferencia [IC 95 %] |
|---|---|---|---|
| Transformer pre-entrenado | 0.725 | 0.661 | +0.064 [+0.053, +0.074] * |
| Transformer sin pre-entrenar | 0.639 | 0.589 | +0.049 [+0.041, +0.057] * |
| GRU | 0.702 | 0.597 | +0.105 [+0.091, +0.118] * |
| FFNN | 0.687 | 0.624 | +0.064 [+0.055, +0.073] * |
| Regresión logística | 0.685 | 0.624 | +0.061 [+0.051, +0.072] * |

### T4: con y sin atributos de último valor (AUC macro)

| | con todos | sin último valor | diferencia [IC 95 %] |
|---|---|---|---|
| Transformer pre-entrenado | 0.772 | 0.749 | +0.023 [+0.019, +0.028] * |
| Transformer sin pre-entrenar | 0.686 | 0.673 | +0.013 [+0.010, +0.016] * |
| GRU | 0.746 | 0.699 | +0.047 [+0.042, +0.053] * |
| FFNN | 0.692 | 0.673 | +0.018 [+0.013, +0.023] * |
| Regresión logística | 0.691 | 0.641 | +0.050 [+0.043, +0.056] * |

### T1, Transformer pre-entrenado: por continuidad de la relación principal (AUL)

| | con todos | sin último valor | diferencia [IC 95 %] |
|---|---|---|---|
| continúa después del corte (n = 17,611) | 0.618 | 0.607 | +0.011 [+0.004, +0.017] * |
| terminó en 2015-12 o antes (n = 726) | 0.579 | 0.538 | +0.041 [+0.018, +0.064] * |

### T1, FFNN: por continuidad de la relación principal (AUL)

| | con todos | sin último valor | diferencia [IC 95 %] |
|---|---|---|---|
| continúa después del corte (n = 17,611) | 0.606 | 0.581 | +0.026 [+0.020, +0.031] * |
| terminó en 2015-12 o antes (n = 726) | 0.573 | 0.574 | -0.001 [-0.019, +0.016] |

### T2, Transformer pre-entrenado: por continuidad de la relación principal (QWK)

| | con todos | sin último valor | diferencia [IC 95 %] |
|---|---|---|---|
| continúa después del corte (n = 14,769) | 0.835 | 0.827 | +0.008 [+0.005, +0.010] * |
| terminó en 2015-12 o antes (n = 481) | 0.653 | 0.656 | -0.003 [-0.032, +0.024] |

### T2, FFNN: por continuidad de la relación principal (QWK)

| | con todos | sin último valor | diferencia [IC 95 %] |
|---|---|---|---|
| continúa después del corte (n = 14,769) | 0.770 | 0.758 | +0.012 [+0.008, +0.015] * |
| terminó en 2015-12 o antes (n = 481) | 0.595 | 0.582 | +0.013 [-0.015, +0.038] |

### T3, Transformer pre-entrenado: por continuidad de la relación principal (AUL)

| | con todos | sin último valor | diferencia [IC 95 %] |
|---|---|---|---|
| continúa después del corte (n = 14,832) | 0.713 | 0.649 | +0.063 [+0.052, +0.074] * |
| terminó en 2015-12 o antes (n = 484) | 0.618 | 0.551 | +0.067 [+0.042, +0.091] * |

### T3, FFNN: por continuidad de la relación principal (AUL)

| | con todos | sin último valor | diferencia [IC 95 %] |
|---|---|---|---|
| continúa después del corte (n = 14,832) | 0.689 | 0.624 | +0.066 [+0.056, +0.075] * |
| terminó en 2015-12 o antes (n = 484) | 0.536 | 0.521 | +0.015 [-0.004, +0.036] |

### T4, Transformer pre-entrenado: por continuidad de la relación principal (AUC macro)

| | con todos | sin último valor | diferencia [IC 95 %] |
|---|---|---|---|
| continúa después del corte (n = 14,833) | 0.769 | 0.744 | +0.025 [+0.021, +0.029] * |
| terminó en 2015-12 o antes (n = 484) | 0.751 | 0.742 | +0.009 [-0.010, +0.028] |

### T4, FFNN: por continuidad de la relación principal (AUC macro)

| | con todos | sin último valor | diferencia [IC 95 %] |
|---|---|---|---|
| continúa después del corte (n = 14,833) | 0.693 | 0.673 | +0.020 [+0.014, +0.026] * |
| terminó en 2015-12 o antes (n = 484) | 0.612 | 0.630 | -0.019 [-0.037, -0.000] * |
