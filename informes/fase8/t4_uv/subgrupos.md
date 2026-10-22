auc_macro por subgrupo (media entre semillas).

| Modelo | mujeres (n=4,951) | varones (n=10,366) | 25-34 (n=5,794) | 35-44 (n=5,728) | 45-55 (n=3,795) |
|---|---|---|---|---|---|
| Transformer pre-entrenado (L2V) | 0.744 | 0.751 | 0.742 | 0.736 | 0.750 |
| Transformer sin pre-entrenar | 0.677 | 0.673 | 0.670 | 0.650 | 0.639 |
| GRU | 0.692 | 0.704 | 0.691 | 0.679 | 0.690 |
| FFNN | 0.664 | 0.673 | 0.664 | 0.648 | 0.661 |
| Regresión logística | 0.642 | 0.638 | 0.636 | 0.617 | 0.622 |
| Logística con estado actual | 0.684 | 0.679 | 0.684 | 0.661 | 0.653 |
| Tabla de vida (edad y sexo) | 0.592 | 0.570 | 0.544 | 0.547 | 0.533 |
