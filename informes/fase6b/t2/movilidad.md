QWK y MAE (media entre semillas) según el trabajador cambie o no de quintil entre 2015 y 2019.

| Modelo | cambian de quintil (n=5,641): QWK / MAE | mantienen (n=9,609): QWK / MAE |
|---|---|---|
| Transformer pre-entrenado (L2V) | 0.480 / 1.030 | 0.977 / 0.092 |
| Transformer sin pre-entrenar | 0.448 / 1.021 | 0.877 / 0.398 |
| GRU | 0.476 / 1.005 | 0.974 / 0.104 |
| FFNN | 0.459 / 1.007 | 0.900 / 0.308 |
| Regresión logística | 0.443 / 1.025 | 0.897 / 0.296 |
| Logística con estado actual | 0.396 / 1.258 | 1.000 / 0.000 |
| Tabla de vida (edad y sexo) | -0.008 / 1.840 | 0.210 / 1.581 |
| Persistencia | 0.396 / 1.258 | 1.000 / 0.000 |
| Mayoría | 0.000 / 1.302 | 0.000 / 1.451 |
| Azar | -0.030 / 1.565 | -0.012 / 1.654 |
