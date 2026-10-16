# Subplan: fase 2 (tokenización, corpus y carga de datos)

Estado: borrador para aprobación (2026-10-16). Complementa [PLAN.md](PLAN.md), secciones 3 y 6.

**Objetivo**: pasar del panel mensual (`data/processed/panel.parquet`) a secuencias que el backbone pueda consumir. Incluye vocabulario, documentos por trabajador, augmentación, enmascarado MLM y SOP, y un DataModule con lotes de largo variable. La fase termina cuando un modelo chico corre en CPU con lotes reales.

## Lo que aprendimos del código de referencia

1. **Tokens especiales**: `life2vec-light` usa este orden: `[PAD]=0, [CLS]=1, [SEP]=2, [MASK]=3, [PLCH0..4]=4..8, [UNK]=9`. Los `ignore_index = [0,4,5,6,7,8]` del backbone son `[PAD]` y los 5 placeholders. **Manteniendo ese orden, no hay que tocarlos.**
2. **Background**: `embeddings.py` anula edad y posición en los primeros 5 tokens (`[CLS]` + 3 de fondo + `[SEP]`). El nuestro tiene 2 tokens de fondo (sexo, año de nacimiento), es decir 4 posiciones. Hay que **parametrizarlo** (`n_tokens_fondo`); si no, el primer token del primer evento pierde su información temporal.
3. **El modelo no tiene embedding de posición por índice**: el único orden que ve es el tiempo (`abspos`, edad) y el segmento. El orden de los tokens sólo importa para las cabezas de atención local (ventana de 38 tokens) y para la tarea SOP. Por eso conviene un **orden fijo y cronológico** de los tokens.
4. **Augmentación de `light` que no conviene copiar tal cual**:
   - `resample_document` corre todas las posiciones para que el primer evento quede en el día 1, y así borra el calendario. En la MLER el calendario importa (crisis de 2002, COVID).
   - `add_noise2time` recorta el tiempo en 4329 días (el período danés). Nuestro período tiene ~9500.
   - `shuffle_sentences` mezcla el orden de los eventos, no los tokens dentro de cada evento (el docstring dice otra cosa). Con nuestro orden fijo no se usa.
5. **Segmentos**: `light` alterna `[2,3,1]` por evento, con el background siempre en 1. Se mantiene.

## 2.1 Diseño: corpus de eventos + tokenizador configurable

Para no materializar una copia del corpus por cada variante (A/B, r34/r32, umbrales), se guarda **un corpus de eventos con códigos crudos** y la conversión a tokens se hace al cargar cada secuencia, con tablas de búsqueda. Las variantes del plan pasan a ser opciones de configuración, no archivos distintos.

**Corpus de eventos** (`data/processed/corpus/eventos_{train,val,test}.parquet`): una fila por trabajador, con listas por evento en orden cronológico:

| Columna | Tipo | Contenido |
|---|---|---|
| `id_trabajador` | int32 | |
| `sexo`, `fnacim` | int8, int16 | background |
| `mes_idx` | list[int16] | mes de cada evento (0 = 1996-01) |
| `pct_rem` | list[int8] | 0-100 (0 = remuneración 0) |
| `tramo` | list[int8] | 1-4, nulo si falta |
| `r34` | list[int16] | código de rama, nulo si falta |
| `provi` | list[int8] | |
| `antig` | list[int8] | `tr_anio_ini` |
| `id_relacion` | list[int16] | para saber si un evento abre una relación o sigue una brecha |
| `nueva` | list[bool] | primer mes de la relación o primer mes después de una brecha |

Son ~8 bytes por evento: ~400 MB sin comprimir para los 49 M eventos y bastante menos en parquet. Entra en la RAM de un nodo de Mendieta (64 GB) y se carga entero en el Dataset.

**Orden de los eventos**: por `mes_idx`; dentro del mismo mes, por inicio de la relación y después por `id_relacion` (que no es cronológico, ver la fase 1).

## 2.2 Vocabulario

Se ajusta **sólo con train** y se guarda en `data/processed/corpus/vocabulario.parquet` (token, id, categoría, frecuencia en train).

| Categoría | Tokens | Cantidad aprox. |
|---|---|---|
| Especiales | `[PAD] [CLS] [SEP] [MASK] [PLCH0-4] [UNK]` (orden de `light`) | 10 |
| Sexo | `SEXO_M`, `SEXO_V` | 2 |
| Año de nacimiento | `NAC_1920` … `NAC_2003`; los extremos poco frecuentes se agrupan (umbral a definir con las frecuencias de train) | ~85 |
| Salario | `REM_0` (remuneración 0), `REM_001` … `REM_100` | 101 |
| Tamaño de empresa | `TAM_1` … `TAM_4` | 4 |
| Rama | `RAMA_xxxx` (r34); los códigos con menos de **50 trabajadores en train** pasan a `RAMA2_xx` (su r32) | ~300 |
| Provincia | `PROV_xx` | 25 |
| Antigüedad de la empresa | `ANT_1` … `ANT_4` | 4 |

Total ≈ 530 tokens, contra 2.043 del paper. Un dato faltante (tramo, rama, provincia) se codifica como `[UNK]`, como en el paper.

## 2.3 Documento por trabajador

- **Background**: `[CLS] SEXO NAC [SEP]`, con segmento 1, `abspos` 0 y edad 0. Son 4 posiciones, que se anulan en el embedding con el nuevo `n_tokens_fondo = 4`.
- **Evento (variante A, línea de base)**: `REM TAM RAMA PROV ANT [SEP]`, 6 posiciones, en orden fijo.
- **Variante B** (opción `eventos_compactos=True`): eventos con `nueva = True` completos; los demás, sólo `REM TAM [SEP]`. Acorta las secuencias casi a la mitad.
- **Posición absoluta**: días desde 1996-01-01 al día 15 del mes (`dias = mes_idx` → fecha), igual que en el panel.
- **Edad**: continua, `anio + (mes − 0,5) / 12 − (fnacim + 0,5)`, porque no hay mes de nacimiento. El paper usa años cumplidos; acá la parte fraccionaria sólo evita saltos de un año entre diciembre y enero. Queda como opción (`edad_entera`) para la ablación.
- **Ventana temporal y truncado**: se cortan los eventos posteriores a `fin_ventana` (ver decisión 1). Si el documento supera `max_length` (2560 por defecto), se descartan los eventos más viejos y se conserva el background, como en el paper.

## 2.4 Tareas de pre-entrenamiento (portadas de `light`, con ajustes)

- **Augmentación** (probabilidades del config del paper, aplicadas en este orden):
  - remuestreo de eventos (p = 0,4; quita del 25 al 50 % de los eventos) **sin correr el calendario**;
  - ruido temporal (p = 0,1; ±15 días, porque la granularidad es mensual);
  - ocultar el background (p = 0,01; 2 tokens `[UNK]`);
  - quitar tokens (p = 0,01).
- **SOP**: 3 clases (orden normal ~90 %, eventos invertidos ~5 %, eventos mezclados ~5 %), con los pesos de clase de `pretrain.py` calculados a partir de esas proporciones en lugar de fijos.
- **MLM**: 30 % de los tokens (80 % `[MASK]`, 10 % sin cambio, 10 % token aleatorio no especial). No se enmascaran `[CLS]`, `[SEP]` ni `[UNK]`.

## 2.5 Carga de datos

- `Dataset` en memoria por split. Cada `__getitem__` arma el documento, aplica augmentación, SOP y MLM, y devuelve arrays numpy.
- **Lotes de largo variable**:
  - `collate` rellena hasta el largo máximo del lote, redondeado a un múltiplo de 64 por la atención local;
  - un `sampler` agrupa secuencias de largo parecido (orden por largo dentro de "mega-lotes" de 50 × tamaño de lote, como hace la práctica habitual de BERT).

  Con una mediana de ~300 tokens contra 2560 de relleno fijo, el pre-entrenamiento debería resultar varias veces más barato que la estimación de la fase 0.
- El tamaño del vector de posiciones MLM (`target_pos`) pasa a depender del lote y no de `max_length`. Hay que ajustar `MaskedLanguageModel` y la tarea MLM.
- `LightningDataModule` con train/val/test y semillas por worker, para que la augmentación sea reproducible.

## 2.6 Cambios en el backbone (`src/labor2vec/modelo/`)

1. Hiperparámetro `n_tokens_fondo` (reemplaza `pos[:, :5]`).
2. Verificar que la atención local de `performer-pytorch` acepte largos variables (múltiplos de 64); si no, ajustar el redondeo del `collate`.
3. Pesos de las clases de SOP desde la configuración de la tarea.
4. Prueba: un lote real de la MLER pasa por el modelo chico en CPU y la pérdida baja en unos pasos.

## 2.7 Construcción y control de memoria

- `python -m labor2vec.tokens.corpus`:
  - procesa el panel **por tramos de `id_trabajador`** (~20 tramos de ~2,5 M filas cada uno), cada uno con DuckDB acotado (un cuarto de la RAM, 4 hilos), y escribe un archivo por tramo y split;
  - así ningún paso tiene en memoria más que una fracción del panel.
- `python -m labor2vec.tokens.vocabulario`: ajusta el vocabulario sobre train.
- **Antes de correr la construcción completa en el contenedor, te pido OK.** Primero corre en una muestra del 1 % para medir tiempo y memoria; el resultado se extrapola.

## 2.8 Pruebas

- **Vocabulario**: determinista, ajustado sólo con train, orden de los especiales compatible con `ignore_index`.
- **Documento**: para un trabajador sintético con dos relaciones simultáneas y una brecha, los tokens, posiciones, edades y segmentos son los esperados (variantes A y B).
- **Corpus**: suma de eventos igual a las filas del panel de los elegibles; orden cronológico; splits sin solapamiento.
- **MLM**: proporción de enmascarados ≈ 30 %; nunca se enmascaran tokens especiales.
- **Largos**: distribución consistente con la estimación del reporte de calidad (mediana ~300 en la variante A).

## Entregables y criterio de aceptación

1. `data/processed/corpus/` (eventos por split + vocabulario), reproducible con un comando.
2. `src/labor2vec/tokens/` (vocabulario, documentos, augmentación, MLM/SOP, Dataset, DataModule) con pruebas en verde.
3. `informes/fase2.md` con:
   - el vocabulario por categoría;
   - la distribución de largos (A y B), con y sin ventana;
   - el rendimiento de carga (secuencias/s por worker), para verificar que alimenta a la GPU (~60 secuencias/s medidas en la fase 0 con 2560 tokens);
   - un lote real que pasa por el modelo en CPU.
4. Sincronizar `data/processed/corpus/` a Mendieta.

## Decisiones que necesito de vos

1. **Ventana del pre-entrenamiento.** El paper pre-entrena sólo con datos anteriores a su corte (2008-2015). Con T1 cortando en 2015-12, pre-entrenar con datos hasta 2021 dejaría que el modelo vea el futuro de los trabajadores de test.
   - **Recomiendo** pre-entrenar con eventos hasta 2015-12, fiel al paper y sin fuga.
   - La historia completa (1996-2021) queda como variante para otros usos.
2. **Población del pre-entrenamiento.** El paper usa la misma población que en sus tareas (25-65 años y ≥12 registros en el último año).
   - **Recomiendo** usar todos los trabajadores elegibles con al menos un evento antes del corte (~460 k en train si la ventana termina en 2015). Son más secuencias para aprender la estructura.
   - El filtro de la cohorte se aplica recién en el fine-tuning de T1.
3. **Variante base**: arrancar con la A (fiel al paper) y dejar la B para las ablaciones de la fase 8. Las dos salen del mismo corpus, así que no cuesta nada tenerlas disponibles.
4. **Umbral de ramas raras**: menos de 50 trabajadores en train → rama a 2 dígitos. Afecta a 19 códigos.
