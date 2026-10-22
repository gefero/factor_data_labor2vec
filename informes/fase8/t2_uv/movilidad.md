QWK y MAE (media entre semillas) según el trabajador cambie o no de quintil entre 2015 y 2019.

| Modelo | cambian de quintil (n=5,641): QWK / MAE | mantienen (n=9,609): QWK / MAE |
|---|---|---|
| Transformer pre-entrenado (L2V) | 0.441 / 1.096 | 0.981 / 0.080 |
| Transformer sin pre-entrenar | 0.442 / 1.115 | 0.978 / 0.088 |
| GRU | 0.442 / 1.127 | 0.983 / 0.069 |
| FFNN | 0.424 / 1.074 | 0.898 / 0.316 |
| Regresión logística | 0.407 / 1.098 | 0.894 / 0.307 |
| Logística con estado actual | 0.396 / 1.258 | 1.000 / 0.000 |
| Tabla de vida (edad y sexo) | -0.008 / 1.840 | 0.210 / 1.581 |
| Persistencia | 0.396 / 1.258 | 1.000 / 0.000 |
| Mayoría | 0.000 / 1.302 | 0.000 / 1.451 |
| Azar | -0.030 / 1.565 | -0.012 / 1.654 |
