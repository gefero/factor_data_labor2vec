auc_macro por subgrupo (media entre semillas).

| Modelo | mujeres (n=4,951) | varones (n=10,366) | 25-34 (n=5,794) | 35-44 (n=5,728) | 45-55 (n=3,795) |
|---|---|---|---|---|---|
| Transformer pre-entrenado (L2V) | 0.770 | 0.773 | 0.760 | 0.760 | 0.789 |
| Transformer sin pre-entrenar | 0.689 | 0.685 | 0.684 | 0.670 | 0.658 |
| GRU | 0.738 | 0.749 | 0.737 | 0.735 | 0.748 |
| FFNN | 0.690 | 0.690 | 0.680 | 0.676 | 0.681 |
| Regresión logística | 0.692 | 0.689 | 0.683 | 0.674 | 0.685 |
| Logística con estado actual | 0.684 | 0.679 | 0.684 | 0.661 | 0.653 |
| Tabla de vida (edad y sexo) | 0.592 | 0.570 | 0.544 | 0.547 | 0.533 |
