auc_macro por subgrupo (media entre semillas).

| Modelo | mujeres (n=503) | varones (n=1,180) | 25-34 (n=930) | 35-44 (n=517) | 45-55 (n=236) |
|---|---|---|---|---|---|
| Transformer pre-entrenado (L2V) | 0.622 | 0.652 | 0.642 | 0.668 | 0.700 |
| Transformer sin pre-entrenar | 0.540 | 0.571 | 0.556 | 0.595 | 0.601 |
| GRU | 0.621 | 0.636 | 0.635 | 0.655 | 0.701 |
| FFNN | 0.576 | 0.615 | 0.617 | 0.641 | 0.668 |
| Regresión logística | 0.579 | 0.602 | 0.608 | 0.619 | 0.640 |
| Logística con estado actual | 0.608 | 0.631 | 0.656 | 0.656 | 0.694 |
| Tabla de vida (edad y sexo) | 0.509 | 0.503 | 0.542 | 0.555 | 0.552 |
