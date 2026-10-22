# Transformer contra líneas de base con variables de tiempo (fase 8b)

Media entre semillas en test; diferencia = transformer menos línea de base (positiva: gana el transformer), con IC al 95 % por bootstrap pareado. `*` = el IC no incluye el 0.

### T1, con todos los atributos: transformer contra líneas de base (AUL)

| | transformer | línea de base | diferencia [IC 95 %] |
|---|---|---|---|
| Regresión logística | 0.619 | 0.617 | +0.002 [-0.005, +0.009] |
| FFNN | 0.619 | 0.608 | +0.010 [+0.003, +0.018] * |
| Regresión logística + tiempo | 0.619 | 0.634 | -0.016 [-0.023, -0.009] * |
| FFNN + tiempo | 0.619 | 0.619 | -0.000 [-0.010, +0.009] |

### T1, sin atributos de último valor: transformer contra líneas de base (AUL)

| | transformer | línea de base | diferencia [IC 95 %] |
|---|---|---|---|
| Regresión logística | 0.606 | 0.585 | +0.021 [+0.013, +0.029] * |
| FFNN | 0.606 | 0.583 | +0.023 [+0.015, +0.032] * |
| Regresión logística + tiempo | 0.606 | 0.618 | -0.011 [-0.020, -0.004] * |
| FFNN + tiempo | 0.606 | 0.620 | -0.013 [-0.023, -0.005] * |

### T2, con todos los atributos: transformer contra líneas de base (QWK)

| | transformer | línea de base | diferencia [IC 95 %] |
|---|---|---|---|
| Regresión logística | 0.830 | 0.758 | +0.071 [+0.064, +0.078] * |
| FFNN | 0.830 | 0.765 | +0.064 [+0.059, +0.070] * |
| Logística con estado actual | 0.830 | 0.822 | +0.007 [+0.005, +0.011] * |
| Regresión logística + tiempo | 0.830 | 0.776 | +0.053 [+0.048, +0.059] * |
| FFNN + tiempo | 0.830 | 0.759 | +0.070 [+0.064, +0.076] * |
| Logística con estado actual + tiempo | 0.830 | 0.821 | +0.008 [+0.005, +0.012] * |

### T2, sin atributos de último valor: transformer contra líneas de base (QWK)

| | transformer | línea de base | diferencia [IC 95 %] |
|---|---|---|---|
| Regresión logística | 0.822 | 0.746 | +0.076 [+0.070, +0.083] * |
| FFNN | 0.822 | 0.753 | +0.069 [+0.063, +0.075] * |
| Logística con estado actual | 0.822 | 0.822 | +0.000 [-0.002, +0.003] |
| Regresión logística + tiempo | 0.822 | 0.767 | +0.056 [+0.050, +0.061] * |
| FFNN + tiempo | 0.822 | 0.771 | +0.051 [+0.046, +0.057] * |
| Logística con estado actual + tiempo | 0.822 | 0.822 | +0.000 [-0.002, +0.003] |

### T3, con todos los atributos: transformer contra líneas de base (AUL)

| | transformer | línea de base | diferencia [IC 95 %] |
|---|---|---|---|
| Regresión logística | 0.725 | 0.685 | +0.040 [+0.029, +0.050] * |
| FFNN | 0.725 | 0.687 | +0.038 [+0.026, +0.048] * |
| Logística con estado actual | 0.725 | 0.656 | +0.069 [+0.057, +0.081] * |
| Regresión logística + tiempo | 0.725 | 0.731 | -0.006 [-0.014, +0.001] |
| FFNN + tiempo | 0.725 | 0.709 | +0.016 [+0.005, +0.025] * |
| Logística con estado actual + tiempo | 0.725 | 0.719 | +0.005 [-0.004, +0.014] |

### T3, sin atributos de último valor: transformer contra líneas de base (AUL)

| | transformer | línea de base | diferencia [IC 95 %] |
|---|---|---|---|
| Regresión logística | 0.661 | 0.624 | +0.037 [+0.025, +0.051] * |
| FFNN | 0.661 | 0.624 | +0.037 [+0.025, +0.051] * |
| Logística con estado actual | 0.661 | 0.656 | +0.005 [-0.008, +0.020] |
| Regresión logística + tiempo | 0.661 | 0.692 | -0.031 [-0.040, -0.020] * |
| FFNN + tiempo | 0.661 | 0.686 | -0.025 [-0.036, -0.014] * |
| Logística con estado actual + tiempo | 0.661 | 0.708 | -0.047 [-0.058, -0.036] * |

### T4, con todos los atributos: transformer contra líneas de base (AUC macro)

| | transformer | línea de base | diferencia [IC 95 %] |
|---|---|---|---|
| Regresión logística | 0.772 | 0.691 | +0.081 [+0.073, +0.090] * |
| FFNN | 0.772 | 0.692 | +0.081 [+0.072, +0.090] * |
| Logística con estado actual | 0.772 | 0.684 | +0.089 [+0.082, +0.095] * |
| Regresión logística + tiempo | 0.772 | 0.722 | +0.051 [+0.043, +0.058] * |
| FFNN + tiempo | 0.772 | 0.699 | +0.074 [+0.065, +0.081] * |
| Logística con estado actual + tiempo | 0.772 | 0.741 | +0.031 [+0.025, +0.038] * |

### T4, sin atributos de último valor: transformer contra líneas de base (AUC macro)

| | transformer | línea de base | diferencia [IC 95 %] |
|---|---|---|---|
| Regresión logística | 0.749 | 0.641 | +0.108 [+0.098, +0.118] * |
| FFNN | 0.749 | 0.673 | +0.076 [+0.066, +0.084] * |
| Logística con estado actual | 0.749 | 0.684 | +0.065 [+0.059, +0.072] * |
| Regresión logística + tiempo | 0.749 | 0.697 | +0.052 [+0.043, +0.060] * |
| FFNN + tiempo | 0.749 | 0.706 | +0.043 [+0.035, +0.050] * |
| Logística con estado actual + tiempo | 0.749 | 0.738 | +0.011 [+0.005, +0.017] * |

### T2C, con todos los atributos: transformer contra líneas de base (AUC macro)

| | transformer | línea de base | diferencia [IC 95 %] |
|---|---|---|---|
| Regresión logística | 0.741 | 0.664 | +0.078 [+0.071, +0.084] * |
| FFNN | 0.741 | 0.647 | +0.094 [+0.088, +0.101] * |
| Logística con estado actual | 0.741 | 0.641 | +0.100 [+0.094, +0.106] * |
| Regresión logística + tiempo | 0.741 | 0.677 | +0.065 [+0.059, +0.071] * |
| FFNN + tiempo | 0.741 | 0.629 | +0.112 [+0.105, +0.120] * |
| Logística con estado actual + tiempo | 0.741 | 0.661 | +0.080 [+0.075, +0.087] * |

### T3D, con todos los atributos: transformer contra líneas de base (AUC macro)

| | transformer | línea de base | diferencia [IC 95 %] |
|---|---|---|---|
| Regresión logística | 0.659 | 0.617 | +0.042 [+0.028, +0.059] * |
| FFNN | 0.659 | 0.632 | +0.028 [+0.014, +0.043] * |
| Logística con estado actual | 0.659 | 0.661 | -0.001 [-0.017, +0.016] |
| Regresión logística + tiempo | 0.659 | 0.619 | +0.040 [+0.025, +0.057] * |
| FFNN + tiempo | 0.659 | 0.581 | +0.078 [+0.058, +0.098] * |
| Logística con estado actual + tiempo | 0.659 | 0.667 | -0.008 [-0.025, +0.010] |

### T4C, con todos los atributos: transformer contra líneas de base (AUL)

| | transformer | línea de base | diferencia [IC 95 %] |
|---|---|---|---|
| Regresión logística | 0.702 | 0.658 | +0.045 [+0.037, +0.052] * |
| FFNN | 0.702 | 0.657 | +0.045 [+0.038, +0.053] * |
| Logística con estado actual | 0.702 | 0.632 | +0.070 [+0.062, +0.079] * |
| Regresión logística + tiempo | 0.702 | 0.688 | +0.014 [+0.009, +0.020] * |
| FFNN + tiempo | 0.702 | 0.675 | +0.027 [+0.021, +0.034] * |
| Logística con estado actual + tiempo | 0.702 | 0.680 | +0.022 [+0.016, +0.028] * |
