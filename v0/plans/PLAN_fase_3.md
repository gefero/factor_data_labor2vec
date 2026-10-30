# Subplan: fase 3 (script de entrenamiento y prueba de humo)

Estado: borrador para aprobación (2026-10-16). Complementa [PLAN.md](PLAN.md), sección 6.

**Objetivo**: tener un script de pre-entrenamiento completo (MLM + SOP, con métricas, checkpoints y reanudación) y comprobar con datos reales que aprende, antes de gastar horas de GPU en la fase 4.

## Punto de partida

Ya se verificó en la fase 2 (con la muestra del 1 %): un lote real pasa por el modelo con la configuración del paper y la pérdida inicial es la esperada (5,94, cerca de ln 519). Pruebas automáticas: el modelo chico baja la pérdida en un lote fijo. **Falta**: entrenar con lotes distintos durante varios cientos de pasos, medir si lo aprendido supera una línea de base, y tener el script que usará la fase 4.

## Lo que falta en el código actual

1. `pretrain.py` (copiado de `light`) **no registra nada**: los pasos devuelven la pérdida total, sin `self.log`, sin pérdidas separadas de MLM y SOP ni métricas.
2. No hay script de entrenamiento, ni configuración (`configs/` está vacío), ni checkpoints, ni reanudación.
3. El calendario OneCycle necesita `epochs × steps_per_epoch` conocidos de antemano; con largo variable y un número fijo de secuencias por época hay que definirlos a partir del tamaño del corpus.

## 3.1 Métricas (cambios en `modelo/pretrain.py`)

Se registran en train (cada 10 pasos) y en validación (por época):

- pérdida total, pérdida MLM y pérdida SOP por separado;
- **exactitud top-1 y perplejidad del MLM** (sobre posiciones enmascaradas, ignorando el relleno);
- exactitud de SOP, y por clase (con 90 % de clase normal, la exactitud global engaña);
- tasa de aprendizaje.

**Línea de base del MLM**: perplejidad de un modelo unigrama con las frecuencias de train del vocabulario (columna `frecuencia`). Si el modelo no la supera, no está aprendiendo estructura. Se calcula una vez y se guarda con el vocabulario.

## 3.2 Script de entrenamiento (`python -m labor2vec.entrenar`)

- **Configuración** en `configs/preentrenamiento.yaml` (hiperparámetros del modelo, opciones del documento, lote, época, precisión, semilla). Opción `--config` y sobrescritura de claves desde la línea de comandos. `vocab_size` y `n_tokens_fondo` se toman del vocabulario y de `documento.py`, no del archivo.
- **Trainer de Lightning**: precisión elegible (`32-true`, `bf16-mixed`), `gradient_clip` opcional, semilla global, sin acumulación de gradientes al principio.
- **Checkpoints**: `last.ckpt` cada N pasos y el mejor por pérdida MLM de validación; sin guardar el optimizador en el "mejor" para que pese poco.
- **Reanudación**: con `--reanudar` retoma desde `last.ckpt` con el mismo comando (necesario para los jobs encadenados de `short`/`multi` de la fase 4).
- **Parada por tiempo**: `--max-minutos` detiene el entrenamiento con un checkpoint antes del límite del job.
- **Registro**: `CSVLogger` (sin dependencias nuevas). TensorBoard queda para después si hace falta.

## 3.3 Definición de la época

El paper usa 30 épocas de 375 pasos con lotes de 80 (30.000 secuencias por época, 900 k en total). Para nuestro corpus (344.025 trabajadores en train con la ventana a 2015-12):

- **Propuesta**: época = una pasada completa por train; `steps_per_epoch = len(sampler)`; lote de 64 (a confirmar con la prueba de memoria de la fase 4). Se mantiene la proporción de ~900 k secuencias vistas, es decir unas 2,6 épocas de pasada completa, pero se parte en épocas más cortas para tener curvas de validación frecuentes.
- La decisión final del calendario se toma en la fase 4 con los tiempos medidos; en la fase 3 sólo se deja el mecanismo parametrizable.

## 3.4 Prueba de humo en CPU (muestra del 1 %)

- **Modelo chico** (`HPARAMS_MINIMO` con el vocabulario real y `n_tokens_fondo=4`): ~500 pasos con lote de 32 sobre `corpus_muestra`. Debe correr en minutos.
- **Modelo del paper**: 20 pasos para comprobar memoria y tiempo por paso en CPU (sólo estructura; no se espera aprendizaje).
- **Criterios**:
  1. la pérdida de entrenamiento baja de forma sostenida (media de los últimos 50 pasos < 80 % de la de los primeros 50);
  2. la perplejidad MLM de validación queda **por debajo de la unigrama**;
  3. sin NaN ni explosión de gradientes;
  4. reanudar desde `last.ckpt` continúa la misma curva (la pérdida del primer paso tras reanudar es comparable a la del último antes de cortar);
  5. el DataModule con `num_workers > 0` produce lotes idénticos entre ejecuciones con la misma semilla.

## 3.5 Prueba corta en Mendieta (opcional, requiere tu aprobación aparte)

Un job en `short` (≤ 1 h, 1 GPU, 10 cores) con **datos reales** para cerrar los pendientes de la fase 0:

- comparar **bf16 contra fp32** en una corrida corta (misma semilla y ~1.000 pasos): ¿converge igual el Performer en bf16?
- medir **pasos/s con largo variable** y confirmar que 10 cores alcanzan para alimentar la GPU;
- medir la memoria pico con lote de 64 y probar si entra un lote mayor;
- con eso se estima el costo real de la fase 4.

No se encola nada sin tu OK explícito (ver la regla de la skill `ccad-unc` para jobs).

## 3.6 Pruebas automáticas

- Métricas: perplejidad y exactitud sobre un caso construido a mano; el relleno no cuenta.
- Línea de base unigrama: coincide con el cálculo manual sobre un vocabulario sintético.
- El script con la muestra sintética del test: corre 5 pasos, escribe `last.ckpt` y el CSV, y reanuda.
- Parada por tiempo: con `--max-minutos` mínimo termina con un checkpoint válido.

## Entregables y criterio de aceptación

1. `src/labor2vec/entrenar.py`, `configs/preentrenamiento.yaml` y los cambios a `pretrain.py`, con pruebas en verde.
2. `informes/fase3.md`: curvas de la prueba de humo (pérdida y perplejidad frente a la unigrama), tiempos en CPU y, si se aprueba 3.5, la comparación bf16 contra fp32 y los tiempos en la A30.
3. Código sincronizado a Mendieta.
4. Cumplir los cinco criterios de 3.4.

## Decisiones que necesito de vos

1. **¿Incluimos la prueba corta en Mendieta (3.5) dentro de la fase 3?**
   - **Recomiendo que sí**, después de que pase la prueba en CPU: es una hora de `short` y evita descubrir un problema de bf16 o de carga en medio de la fase 4.
2. **Época = una pasada completa de train** (3.3), con el calendario final fijado en la fase 4. ¿De acuerdo?
3. **Línea de base de aceptación**: perplejidad MLM por debajo de la unigrama. ¿Querés además otra línea de base (por ejemplo un modelo de bigramas)? **Recomiendo** sólo la unigrama por ahora.
4. **Registro con `CSVLogger`** en vez de TensorBoard. ¿De acuerdo?
