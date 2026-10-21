# Resultados de la fase 7 (generado por `analisis.personas analizar`)

## Mapa de personas

Pureza de los 5 vecinos más cercanos (coseno, submuestra de 4.000 personas del test; entre paréntesis, el valor esperado al azar). Si es mayor que el azar, personas parecidas en ese atributo quedan cerca.

| modelo | y | sexo | letra | tamaño | salario | antigüedad | edad |
|---|---|---|---|---|---|---|---|
| T1 ajustado | 0.742 (0.729) | 0.641 (0.556) | 0.424 (0.205) | 0.614 (0.364) | 0.731 (0.334) | 0.570 (0.333) | 0.559 (0.335) |
| T3 ajustado | 0.827 (0.812) | 0.657 (0.557) | 0.615 (0.203) | 0.597 (0.367) | 0.601 (0.333) | 0.564 (0.333) | 0.573 (0.336) |
| pre-entrenado sin ajustar | 0.734 (0.716) | 0.659 (0.552) | 0.537 (0.205) | 0.679 (0.362) | 0.657 (0.334) | 0.429 (0.333) | 0.416 (0.335) |

## TCAV

Puntaje = fracción de personas para las que mover el estado de todos sus tokens hacia el concepto sube la probabilidad de la clase 1 (0,5 = el concepto no importa; el nulo son direcciones al azar). `exactitud` = qué tan bien separa linealmente el concepto de las demás personas (datos retenidos). p con Bonferroni.

### T1 ajustado, última capa

| concepto | clase real | TCAV | nulo | exactitud | p |
|---|---|---|---|---|---|
| mujer | 0 | 0.987 ± 0.076 | 0.556 | 0.748 | 1.0e-06 * |
| mujer | 1 | 0.981 ± 0.094 | 0.559 | 0.748 | 1.8e-06 * |
| empresa grande (>200) | 0 | 0.000 ± 0.001 | 0.556 | 0.909 | 1.3e-09 * |
| empresa grande (>200) | 1 | 0.001 ± 0.002 | 0.559 | 0.909 | 1.2e-09 * |
| empresa chica (≤10) | 0 | 1.000 ± 0.000 | 0.556 | 0.922 | 4.8e-07 * |
| empresa chica (≤10) | 1 | 1.000 ± 0.001 | 0.559 | 0.922 | 5.8e-07 * |
| salario alto | 0 | 0.000 ± 0.001 | 0.556 | 0.910 | 1.3e-09 * |
| salario alto | 1 | 0.000 ± 0.002 | 0.559 | 0.910 | 1.2e-09 * |
| salario bajo | 0 | 1.000 ± 0.001 | 0.556 | 0.891 | 4.8e-07 * |
| salario bajo | 1 | 0.999 ± 0.003 | 0.559 | 0.891 | 5.9e-07 * |
| antigüedad alta | 0 | 1.000 ± 0.000 | 0.556 | 0.903 | 4.8e-07 * |
| antigüedad alta | 1 | 1.000 ± 0.000 | 0.559 | 0.903 | 5.7e-07 * |
| antigüedad baja | 0 | 0.000 ± 0.000 | 0.556 | 0.908 | 1.3e-09 * |
| antigüedad baja | 1 | 0.000 ± 0.000 | 0.559 | 0.908 | 1.2e-09 * |
| mayor | 0 | 1.000 ± 0.000 | 0.556 | 0.954 | 4.8e-07 * |
| mayor | 1 | 1.000 ± 0.000 | 0.559 | 0.954 | 5.7e-07 * |
| joven | 0 | 0.000 ± 0.000 | 0.556 | 0.939 | 1.3e-09 * |
| joven | 1 | 0.000 ± 0.000 | 0.559 | 0.939 | 1.2e-09 * |
| rama letra 4 | 0 | 0.844 ± 0.311 | 0.556 | 0.861 | 9.0e-03 * |
| rama letra 4 | 1 | 0.875 ± 0.280 | 0.559 | 0.861 | 1.9e-03 * |
| rama letra 7 | 0 | 0.831 ± 0.333 | 0.556 | 0.824 | 1.9e-02 * |
| rama letra 7 | 1 | 0.804 ± 0.358 | 0.559 | 0.824 | 7.0e-02 |
| rama letra 11 | 0 | 1.000 ± 0.001 | 0.556 | 0.818 | 4.8e-07 * |
| rama letra 11 | 1 | 0.999 ± 0.004 | 0.559 | 0.818 | 6.0e-07 * |
| rama letra 9 | 0 | 0.000 ± 0.000 | 0.556 | 0.877 | 1.3e-09 * |
| rama letra 9 | 1 | 0.000 ± 0.000 | 0.559 | 0.877 | 1.2e-09 * |
| rama letra 12 | 0 | 0.015 ± 0.050 | 0.556 | 0.974 | 2.9e-09 * |
| rama letra 12 | 1 | 0.033 ± 0.086 | 0.559 | 0.974 | 6.8e-09 * |
| rama letra 14 | 0 | 1.000 ± 0.001 | 0.556 | 0.849 | 4.8e-07 * |
| rama letra 14 | 1 | 1.000 ± 0.001 | 0.559 | 0.849 | 5.8e-07 * |

### T1 ajustado, penúltima capa

| concepto | clase real | TCAV | nulo | exactitud | p |
|---|---|---|---|---|---|
| mujer | 0 | 0.999 ± 0.002 | 0.515 | 0.747 | 1.5e-09 * |
| mujer | 1 | 0.998 ± 0.007 | 0.516 | 0.747 | 3.9e-09 * |
| empresa grande (>200) | 0 | 0.020 ± 0.099 | 0.515 | 0.907 | 8.6e-10 * |
| empresa grande (>200) | 1 | 0.028 ± 0.132 | 0.516 | 0.907 | 3.3e-09 * |
| empresa chica (≤10) | 0 | 1.000 ± 0.000 | 0.515 | 0.917 | 1.5e-09 * |
| empresa chica (≤10) | 1 | 1.000 ± 0.000 | 0.516 | 0.917 | 3.5e-09 * |
| salario alto | 0 | 0.000 ± 0.000 | 0.515 | 0.906 | 2.6e-10 * |
| salario alto | 1 | 0.000 ± 0.000 | 0.516 | 0.906 | 5.4e-10 * |
| salario bajo | 0 | 1.000 ± 0.000 | 0.515 | 0.885 | 1.5e-09 * |
| salario bajo | 1 | 1.000 ± 0.000 | 0.516 | 0.885 | 3.5e-09 * |
| antigüedad alta | 0 | 1.000 ± 0.000 | 0.515 | 0.905 | 1.5e-09 * |
| antigüedad alta | 1 | 1.000 ± 0.000 | 0.516 | 0.905 | 3.5e-09 * |
| antigüedad baja | 0 | 0.000 ± 0.000 | 0.515 | 0.914 | 2.6e-10 * |
| antigüedad baja | 1 | 0.000 ± 0.000 | 0.516 | 0.914 | 5.4e-10 * |
| mayor | 0 | 0.996 ± 0.005 | 0.515 | 0.964 | 1.9e-09 * |
| mayor | 1 | 0.989 ± 0.012 | 0.516 | 0.964 | 6.8e-09 * |
| joven | 0 | 0.003 ± 0.006 | 0.515 | 0.950 | 3.1e-10 * |
| joven | 1 | 0.010 ± 0.013 | 0.516 | 0.950 | 9.5e-10 * |
| rama letra 4 | 0 | 0.766 ± 0.233 | 0.515 | 0.868 | 5.9e-03 * |
| rama letra 4 | 1 | 0.862 ± 0.180 | 0.516 | 0.868 | 2.8e-05 * |
| rama letra 7 | 0 | 0.993 ± 0.020 | 0.515 | 0.812 | 2.3e-09 * |
| rama letra 7 | 1 | 0.983 ± 0.042 | 0.516 | 0.812 | 9.3e-09 * |
| rama letra 11 | 0 | 0.999 ± 0.005 | 0.515 | 0.808 | 1.6e-09 * |
| rama letra 11 | 1 | 0.996 ± 0.013 | 0.516 | 0.808 | 4.4e-09 * |
| rama letra 9 | 0 | 0.000 ± 0.000 | 0.515 | 0.880 | 2.6e-10 * |
| rama letra 9 | 1 | 0.000 ± 0.000 | 0.516 | 0.880 | 5.4e-10 * |
| rama letra 12 | 0 | 0.017 ± 0.028 | 0.515 | 0.974 | 6.8e-10 * |
| rama letra 12 | 1 | 0.039 ± 0.054 | 0.516 | 0.974 | 5.1e-09 * |
| rama letra 14 | 0 | 0.924 ± 0.196 | 0.515 | 0.851 | 4.3e-07 * |
| rama letra 14 | 1 | 0.898 ± 0.241 | 0.516 | 0.851 | 8.2e-06 * |

### T3 ajustado, última capa

| concepto | clase real | TCAV | nulo | exactitud | p |
|---|---|---|---|---|---|
| mujer | 0 | 0.235 ± 0.363 | 0.467 | 0.733 | 9.9e-02 |
| mujer | 1 | 0.232 ± 0.368 | 0.471 | 0.733 | 7.9e-02 |
| empresa grande (>200) | 0 | 0.479 ± 0.321 | 0.467 | 0.903 | 1.0e+00 |
| empresa grande (>200) | 1 | 0.707 ± 0.285 | 0.471 | 0.903 | 4.2e-02 * |
| empresa chica (≤10) | 0 | 0.630 ± 0.333 | 0.467 | 0.918 | 7.0e-01 |
| empresa chica (≤10) | 1 | 0.431 ± 0.357 | 0.471 | 0.918 | 1.0e+00 |
| salario alto | 0 | 0.999 ± 0.008 | 0.467 | 0.898 | 1.9e-09 * |
| salario alto | 1 | 0.999 ± 0.008 | 0.471 | 0.898 | 2.1e-09 * |
| salario bajo | 0 | 0.132 ± 0.235 | 0.467 | 0.879 | 3.1e-04 * |
| salario bajo | 1 | 0.220 ± 0.290 | 0.471 | 0.879 | 2.5e-02 * |
| antigüedad alta | 0 | 0.995 ± 0.008 | 0.467 | 0.898 | 2.3e-09 * |
| antigüedad alta | 1 | 0.999 ± 0.001 | 0.471 | 0.898 | 2.1e-09 * |
| antigüedad baja | 0 | 0.002 ± 0.004 | 0.467 | 0.907 | 7.1e-08 * |
| antigüedad baja | 1 | 0.000 ± 0.001 | 0.471 | 0.907 | 5.0e-08 * |
| mayor | 0 | 0.544 ± 0.213 | 0.467 | 0.962 | 1.0e+00 |
| mayor | 1 | 0.827 ± 0.148 | 0.471 | 0.962 | 4.0e-05 * |
| joven | 0 | 0.646 ± 0.288 | 0.467 | 0.951 | 3.4e-01 |
| joven | 1 | 0.452 ± 0.274 | 0.471 | 0.951 | 1.0e+00 |
| rama letra 4 | 0 | 0.994 ± 0.019 | 0.467 | 0.878 | 2.5e-09 * |
| rama letra 4 | 1 | 0.999 ± 0.002 | 0.471 | 0.878 | 2.1e-09 * |
| rama letra 7 | 0 | 0.814 ± 0.246 | 0.467 | 0.844 | 1.9e-04 * |
| rama letra 7 | 1 | 0.648 ± 0.325 | 0.471 | 0.844 | 4.4e-01 |
| rama letra 11 | 0 | 1.000 ± 0.000 | 0.467 | 0.874 | 1.8e-09 * |
| rama letra 11 | 1 | 1.000 ± 0.000 | 0.471 | 0.874 | 2.0e-09 * |
| rama letra 9 | 0 | 0.000 ± 0.001 | 0.467 | 0.906 | 6.5e-08 * |
| rama letra 9 | 1 | 0.001 ± 0.004 | 0.471 | 0.906 | 5.1e-08 * |
| rama letra 12 | 0 | 0.013 ± 0.069 | 0.467 | 0.979 | 1.4e-07 * |
| rama letra 12 | 1 | 0.032 ± 0.121 | 0.471 | 0.979 | 3.7e-07 * |
| rama letra 14 | 0 | 0.998 ± 0.005 | 0.467 | 0.870 | 2.0e-09 * |
| rama letra 14 | 1 | 1.000 ± 0.001 | 0.471 | 0.870 | 2.0e-09 * |

### T3 ajustado, penúltima capa

| concepto | clase real | TCAV | nulo | exactitud | p |
|---|---|---|---|---|---|
| mujer | 0 | 0.079 ± 0.140 | 0.488 | 0.729 | 1.4e-07 * |
| mujer | 1 | 0.174 ± 0.201 | 0.476 | 0.729 | 1.7e-04 * |
| empresa grande (>200) | 0 | 0.116 ± 0.224 | 0.488 | 0.906 | 4.5e-06 * |
| empresa grande (>200) | 1 | 0.190 ± 0.236 | 0.476 | 0.906 | 6.9e-04 * |
| empresa chica (≤10) | 0 | 0.993 ± 0.016 | 0.488 | 0.916 | 2.4e-10 * |
| empresa chica (≤10) | 1 | 0.970 ± 0.064 | 0.476 | 0.916 | 3.0e-10 * |
| salario alto | 0 | 0.957 ± 0.064 | 0.488 | 0.898 | 2.2e-09 * |
| salario alto | 1 | 0.877 ± 0.164 | 0.476 | 0.898 | 1.9e-07 * |
| salario bajo | 0 | 0.516 ± 0.353 | 0.488 | 0.880 | 1.0e+00 |
| salario bajo | 1 | 0.638 ± 0.337 | 0.476 | 0.880 | 4.7e-01 |
| antigüedad alta | 0 | 0.992 ± 0.029 | 0.488 | 0.898 | 2.6e-10 * |
| antigüedad alta | 1 | 0.998 ± 0.006 | 0.476 | 0.898 | 5.5e-11 * |
| antigüedad baja | 0 | 0.007 ± 0.025 | 0.488 | 0.911 | 1.1e-09 * |
| antigüedad baja | 1 | 0.013 ± 0.060 | 0.476 | 0.911 | 2.1e-09 * |
| mayor | 0 | 0.577 ± 0.220 | 0.488 | 0.968 | 1.0e+00 |
| mayor | 1 | 0.843 ± 0.141 | 0.476 | 0.968 | 1.3e-06 * |
| joven | 0 | 0.502 ± 0.309 | 0.488 | 0.955 | 1.0e+00 |
| joven | 1 | 0.393 ± 0.282 | 0.476 | 0.955 | 1.0e+00 |
| rama letra 4 | 0 | 0.628 ± 0.330 | 0.488 | 0.873 | 9.4e-01 |
| rama letra 4 | 1 | 0.790 ± 0.237 | 0.476 | 0.873 | 1.4e-04 * |
| rama letra 7 | 0 | 1.000 ± 0.001 | 0.488 | 0.828 | 1.7e-10 * |
| rama letra 7 | 1 | 0.998 ± 0.009 | 0.476 | 0.828 | 5.6e-11 * |
| rama letra 11 | 0 | 1.000 ± 0.000 | 0.488 | 0.845 | 1.7e-10 * |
| rama letra 11 | 1 | 1.000 ± 0.000 | 0.476 | 0.845 | 4.9e-11 * |
| rama letra 9 | 0 | 0.098 ± 0.084 | 0.488 | 0.895 | 3.2e-07 * |
| rama letra 9 | 1 | 0.016 ± 0.020 | 0.476 | 0.895 | 2.5e-09 * |
| rama letra 12 | 0 | 0.000 ± 0.000 | 0.488 | 0.979 | 7.3e-10 * |
| rama letra 12 | 1 | 0.001 ± 0.002 | 0.476 | 0.979 | 9.6e-10 * |
| rama letra 14 | 0 | 0.693 ± 0.196 | 0.488 | 0.858 | 3.3e-02 * |
| rama letra 14 | 1 | 0.904 ± 0.089 | 0.476 | 0.858 | 1.9e-08 * |

## Saliency (SmoothGrad)

Participación en el puntaje total / participación en la cantidad de tokens. Un cociente mayor que 1 quiere decir que ese tipo de token pesa más de lo que le tocaría por su cantidad.

### T1 ajustado, personas con y = 1

| categoría de token | % del puntaje | % de los tokens | cociente |
|---|---|---|---|
| especial | 29.6 | 16.9 | 1.76 |
| antigüedad | 23.7 | 16.6 | 1.43 |
| tamaño | 13.1 | 16.6 | 0.79 |
| provincia | 12.5 | 16.6 | 0.75 |
| salario | 11.9 | 16.6 | 0.72 |
| rama | 9.0 | 16.6 | 0.54 |
| sexo | 0.1 | 0.1 | 0.64 |
| nacimiento | 0.1 | 0.1 | 0.46 |
| rama2 | 0.0 | 0.0 | 0.11 |

| meses antes del corte (eventos) | % del puntaje | % de los tokens | cociente |
|---|---|---|---|
| 0-3 | 6.5 | 3.6 | 1.80 |
| 3-6 | 3.7 | 1.8 | 2.01 |
| 6-12 | 11.6 | 5.5 | 2.10 |
| 12-24 | 18.1 | 10.1 | 1.79 |
| 24-60 | 29.7 | 24.8 | 1.20 |
| 60+ | 30.4 | 54.1 | 0.56 |

| evento | puntaje medio por token | tokens |
|---|---|---|
| con cambio | 1.48e-03 | 25080 |
| sin cambio | 1.47e-03 | 647244 |

### T1 ajustado, personas con y = 0

| categoría de token | % del puntaje | % de los tokens | cociente |
|---|---|---|---|
| especial | 29.5 | 16.8 | 1.75 |
| antigüedad | 23.9 | 16.6 | 1.44 |
| tamaño | 13.2 | 16.6 | 0.79 |
| provincia | 12.4 | 16.6 | 0.75 |
| salario | 11.8 | 16.6 | 0.71 |
| rama | 9.1 | 16.6 | 0.55 |
| sexo | 0.1 | 0.1 | 0.58 |
| nacimiento | 0.0 | 0.1 | 0.39 |
| rama2 | 0.0 | 0.0 | 0.99 |

| meses antes del corte (eventos) | % del puntaje | % de los tokens | cociente |
|---|---|---|---|
| 0-3 | 5.4 | 3.2 | 1.70 |
| 3-6 | 3.0 | 1.6 | 1.88 |
| 6-12 | 9.5 | 4.8 | 1.98 |
| 12-24 | 15.9 | 9.0 | 1.77 |
| 24-60 | 30.8 | 23.9 | 1.29 |
| 60+ | 35.4 | 57.5 | 0.61 |

| evento | puntaje medio por token | tokens |
|---|---|---|
| con cambio | 1.22e-03 | 28536 |
| sin cambio | 1.25e-03 | 763812 |

### T3 ajustado, personas con y = 1

| categoría de token | % del puntaje | % de los tokens | cociente |
|---|---|---|---|
| especial | 31.1 | 16.9 | 1.84 |
| antigüedad | 23.4 | 16.6 | 1.41 |
| tamaño | 15.0 | 16.6 | 0.91 |
| provincia | 12.6 | 16.6 | 0.76 |
| rama | 9.2 | 16.6 | 0.56 |
| salario | 8.5 | 16.6 | 0.52 |
| sexo | 0.1 | 0.1 | 0.55 |
| nacimiento | 0.1 | 0.1 | 0.39 |

| meses antes del corte (eventos) | % del puntaje | % de los tokens | cociente |
|---|---|---|---|
| 0-3 | 6.9 | 3.8 | 1.82 |
| 3-6 | 3.8 | 1.9 | 2.02 |
| 6-12 | 11.6 | 5.6 | 2.06 |
| 12-24 | 18.7 | 10.4 | 1.80 |
| 24-60 | 31.3 | 25.8 | 1.21 |
| 60+ | 27.8 | 52.5 | 0.53 |

| evento | puntaje medio por token | tokens |
|---|---|---|
| con cambio | 1.48e-03 | 24396 |
| sin cambio | 1.49e-03 | 504168 |

### T3 ajustado, personas con y = 0

| categoría de token | % del puntaje | % de los tokens | cociente |
|---|---|---|---|
| especial | 30.8 | 16.8 | 1.83 |
| antigüedad | 23.3 | 16.6 | 1.40 |
| tamaño | 15.0 | 16.6 | 0.91 |
| provincia | 12.5 | 16.6 | 0.75 |
| rama | 9.6 | 16.6 | 0.58 |
| salario | 8.6 | 16.6 | 0.52 |
| sexo | 0.1 | 0.1 | 0.53 |
| nacimiento | 0.0 | 0.1 | 0.40 |

| meses antes del corte (eventos) | % del puntaje | % de los tokens | cociente |
|---|---|---|---|
| 0-3 | 6.0 | 3.1 | 1.95 |
| 3-6 | 3.3 | 1.5 | 2.14 |
| 6-12 | 10.2 | 4.6 | 2.21 |
| 12-24 | 16.8 | 8.8 | 1.92 |
| 24-60 | 31.3 | 23.4 | 1.34 |
| 60+ | 32.4 | 58.7 | 0.55 |

| evento | puntaje medio por token | tokens |
|---|---|---|
| con cambio | 1.16e-03 | 25656 |
| sin cambio | 1.22e-03 | 791634 |
