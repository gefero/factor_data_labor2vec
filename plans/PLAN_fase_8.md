# Subplan: fase 8 (ablaciones, controles y informe final)

Estado: aprobado (2026-10-21). Complementa [PLAN.md](PLAN.md) (sección 7 y hoja de ruta fila 8).

## Contexto

Las fases 6, 6b y 7 dejaron tres dudas abiertas sobre qué mide realmente el modelo:

1. **¿Hay fuga de información posterior al corte?** `provi`, rama y `tr_anio_ini` son atributos de "último valor" (los vigentes en el último dato de la relación, no mes a mes). En la fase 7 el token de antigüedad pesó 1,4 veces lo que le toca por cantidad. No se sabe si eso es fuga o información legítima.
2. **¿T3 y T4 dependen del momento y el orden de los eventos?** Es la hipótesis de la fase 6b (por eso el transformer le gana a los baselines de conteo en esas dos tareas). El saliency de la fase 7 no encontró evidencia, pero su detector de cambios era burdo.
3. **¿Los resultados de T2-T4 dependen de cómo se definió la tarea?** Sólo se probó una definición de cada una.

Más el **informe final** del proyecto, que consolida las fases 0 a 8.

## Alcance (decidido con el usuario)

Incluido:
- Control temporal (orden y momento de los eventos), sin pre-entrenar de nuevo.
- Variantes de T2, T3 y T4, sin pre-entrenar de nuevo.
- Variante **sin atributos de último valor**, con pre-entrenamiento nuevo.
- Informe de la fase 8 e informe final.

Queda fuera (por decisión del usuario; se puede retomar después): variante B de eventos compactos, `r32` contra `r34`, cortes múltiples y extras estadísticos de T1-T4.

Semillas: 1 pre-entrenamiento nuevo; en los reajustes, 3 semillas del transformer pre-entrenado y 1 de las demás (sin pre-entrenar y GRU).

## 1. Opciones del documento (código común)

En `src/labor2vec/tokens/documento.py`, dos opciones nuevas en `Opciones`, con valor por defecto igual al comportamiento actual:

- `atributos: str = "todos"`. Con `"sin_ultimo_valor"`, cada evento es `REM TAM [SEP]` (3 tokens), sin rama, provincia ni antigüedad. Se implementa en `bloques_de_eventos` (que ya arma la variante B con 3 columnas) y en `DatasetPretrain.largos` (`tokens/datos.py`, que hoy sólo conoce 6 tokens por evento o la variante B). El vocabulario no cambia: los tokens sin uso quedan con su valor inicial.
- `control_temporal: str | None = None`:
  - `"tiempo"`: se permutan entre los eventos de la misma persona los días absolutos y la edad, así el modelo ve los mismos eventos en el mismo orden pero sin información de cuándo ocurrió cada uno. Es el control que de verdad quita la información temporal, porque el modelo no tiene posición por índice: el orden sólo le llega por `abspos`.
  - `"orden"`: se permuta el orden de los eventos en la secuencia, con su fecha. Prueba si el modelo usa el orden de la secuencia (atención local y ciclo de segmentos).
  El sorteo usa el mismo generador determinista por trabajador que ya usan val y test (`DatasetPretrain._generador`), así que es reproducible.
- Una función `opciones_de(d: dict) -> Opciones` en `documento.py`, para armar las `Opciones` desde la sección `datos` de un YAML. Hoy ese armado está repetido en `entrenar.preparar`, `ajustar.preparar` y `analisis/extraer.main`, y `tareas/tabulares.cargar` ni siquiera pasa `eventos_compactos`. Se reemplaza en los cuatro lugares.

## 2. Control temporal sin pre-entrenar de nuevo

- **Nuevo `src/labor2vec/tareas/predecir.py`** (corre en el cluster, sin polars): carga un `modelo_final.pt` ajustado (como `analisis/extraer.cargar_modelo`), arma el `DataModuleT1` con `--set datos.control_temporal=tiempo|orden` y escribe `predicciones_{val,test}.parquet` con `ajustar.escribir_predicciones`. No entrena.
- Se aplica a las 3 semillas del transformer ya ajustado de T1, T2, T3 y T4 (`t{1..4}_l2v_s{0,1,2}` en el cluster), con los dos controles: 24 pasadas de ~2 minutos, agrupadas en un job por tarea (`scripts/cluster/predecir.sbatch`, copia de `personas.sbatch`).
- **Reajuste con el tiempo destruido** en T3 y T4 (donde se espera que importe): `ajustar` con `--set datos.control_temporal=tiempo`, 3 semillas, desde el modelo pre-entrenado `pre_150ep`. Son 6 jobs de ~30 minutos. Responde si la tarea *necesita* la información temporal, no sólo si este modelo la usa.
- Lectura: si la métrica casi no cambia, el momento de los eventos no importa y la hipótesis de la fase 6b es falsa. Si cae, importa. Sólo predecir con el control mide cuánto se apoya el modelo en el tiempo; el reajuste mide si la tarea lo necesita.

## 3. Variantes de T2, T3 y T4

En `src/labor2vec/tareas/derivadas.py`, `tablas()` pasa a devolver tres tablas más (mismos splits y cohortes de T2-T4), cada una en `data/processed/tareas/`:

| Tarea | Archivo | Cohorte | Etiqueta `y` | Pérdida / monitor |
|---|---|---|---|---|
| T2c: cambio de quintil | `t2c.parquet` | la de T2 | 0 baja, 1 igual, 2 sube (quintil de 2019 contra el de 2015), ordinal | `ordinal` / `auc_macro` |
| T3d: destino de la letra | `t3d.parquet` | quienes cambian de letra en T3 (~11 % de 102.885) | letra de 2019: las 5 más frecuentes en train más «otras» (6 clases, nominal) | `balanceada` / `auc_macro` |
| T4c: cambio de tamaño sí/no | `t4c.parquet` | la de T4 | 1 si `tramo2019 != tramo2015` (~19 %), binaria | `balanceada` / `aul` |

- Configs `configs/ajuste_{t2c,t3d,t4c}.yaml` y `gru_*` (copias de las de T2-T4 con `n_clases`, tabla y monitor cambiados).
- Reajuste de cada una: transformer pre-entrenado × 3 semillas, sin pre-entrenar × 1 y GRU × 1 (5 jobs por tarea, 15 en total), más las líneas de base tabulares (`tareas/tabulares.py` con `--tarea` ampliado: tabla de vida, logística, logística con estado de 2015 y FFNN).
- `tareas/evaluar.py`: se agregan las tres tareas a `CONFIG`. T3d es nominal, así que sin QWK ni MAE (AUC macro, exactitud balanceada y F1 macro). T2c queda con persistencia = «igual» como línea de base.
- Cuidado: T3d tiene unas 8.000 personas de train, mucho menos que las otras. Los intervalos van a ser anchos y se dice en el informe.

## 4. Sin atributos de último valor

- **Pre-entrenamiento nuevo** `pre_150ep_sinuv`: misma configuración y semilla 0 que `pre_150ep` (150 épocas × 375 pasos, lote 32×2), con `datos.atributos=sin_ultimo_valor`. Se corre con `scripts/cluster/preentrenar.sbatch` en tramos `short` encadenados con `afterok`, igual que en la fase 4. Con eventos de la mitad del largo debería ser más rápido que las 5 h de la corrida original, pero **no está medido**: antes de encadenar se hace un tramo de 1 h para medir la velocidad y se ajusta la cantidad de tramos.
- **Reajuste** en T1, T2, T3 y T4 con `--set datos.atributos=sin_ultimo_valor --set preentrenado=~/labor2vec/runs/pre_150ep_sinuv/modelo_final.pt`: transformer × 3 semillas, sin pre-entrenar × 1 y GRU × 1 (5 jobs por tarea, 20 en total). Las líneas de base tabulares (logística y FFNN) también se repiten sin esos atributos, para que todos los modelos vean la misma entrada (`tabulares.cargar` acepta `atributos`).
- **Qué mide, y lo que no mide.** La comparación con y sin atributos es una **cota superior** del efecto de la fuga, no una medida limpia: la rama de 2015 es información legítima para predecir un cambio de rama, así que quitarla baja T3 aunque no hubiera fuga. Para separar las dos cosas se agrega un análisis de **subgrupos por continuidad**, sin entrenar nada:
  - Con el panel (`data/processed/panel.parquet`) se marca, para cada persona, si su relación principal de 2015 **continúa después de 2015-12** (ahí el «último valor» puede venir del futuro) o **terminó en 2015-12 o antes** (ahí el valor es el de antes del corte: sin fuga posible).
  - Se calcula la métrica de los modelos con y sin atributos en cada subgrupo (nueva función `subgrupos_continuidad` en `tareas/evaluar.py`, que ya tiene `subgrupos`). Si la ventaja de tener los atributos se concentra en quienes continúan, es señal de fuga. Si se reparte igual, es información legítima.
  - El subgrupo tiene sesgo de selección (quienes terminan una relación en diciembre de 2015 no son una muestra al azar), y se dice en el informe.
- **Diferencia pareada con IC** entre dos conjuntos de predicciones sobre las mismas personas: pequeña función `bootstrap_diferencia` en `tareas/metricas.py`, a partir del `_bootstrap` existente. Se usa sólo para estas comparaciones de ablación.

## 5. Informes

- `informes/fase8.md`: resultados de las tres piezas, con tablas en `informes/fase8/` generadas por `evaluar` y un resumen en lenguaje simple.
- `informes/informe_final.md`: consolidado de las fases 0 a 8 (datos, tokenización, modelo, pre-entrenamiento, espacio de conceptos, tareas, espacio de personas y ablaciones), con los números clave, lo que se confirmó, lo que no, las limitaciones y lo que quedó sin hacer. Pensado para alguien que no siguió el proyecto.

## Costos y etapas (cada etapa con tu OK antes de encolar o correr algo pesado)

| Etapa | Qué | GPU (estimada, no medida salvo lo dicho) |
|---|---|---|
| A | Código, tests y pruebas de humo en CPU. | ninguna |
| B | Tablas de las variantes (local, duckdb); predicciones con control (4 jobs); reajuste con el tiempo destruido (6 jobs); variantes de T2-T4 (15 jobs + tabulares). | ~25 jobs de 15 a 40 min |
| C | Tramo de prueba (1 h) y después pre-entrenamiento `pre_150ep_sinuv` (≤ 5 h; la original llevó 5 h 8 min). | ~3 a 5 h |
| D | Reajustes sin atributos de último valor (20 jobs + tabulares). | ~20 jobs de 15 a 40 min |
| E | Evaluación, informes e informe final. | local |

Las etapas B y C son independientes y pueden correr a la vez. Hasta 6 jobs en paralelo ya se vieron en Mendieta. Todos los `sbatch` con `--mail-type=BEGIN,END,FAIL` al correo del usuario. Los jobs de reajuste reusan `scripts/cluster/ajustar.sbatch`.

## Archivos principales

- Modificar: `tokens/documento.py` (opciones y `opciones_de`), `tokens/datos.py` (`largos`), `entrenar.py`, `ajustar.py`, `analisis/extraer.py`, `tareas/tabulares.py`, `tareas/derivadas.py`, `tareas/evaluar.py`, `tareas/metricas.py`.
- Nuevos: `tareas/predecir.py`, `scripts/cluster/predecir.sbatch`, `configs/ajuste_{t2c,t3d,t4c}.yaml`, `configs/gru_{t2c,t3d,t4c}.yaml`, `tests/test_ablaciones.py`, `plans/PLAN_fase_8.md`, `informes/fase8.md`, `informes/informe_final.md`.

## Verificación

1. `tests/test_ablaciones.py`, con el corpus sintético de `tests/conftest.py`:
   - `sin_ultimo_valor` da 3 tokens por evento y ningún token de rama, provincia ni antigüedad; `largos()` coincide con el largo real del documento;
   - `control_temporal="tiempo"` conserva el mismo multiconjunto de tokens y cambia sólo `abspos` y edad; `"orden"` conserva los pares (token, fecha) y cambia el orden; ambos son deterministas;
   - con las opciones por defecto, los documentos son idénticos a los de antes (regresión);
   - las tablas de T2c, T3d y T4c: etiquetas, cohortes y splits.
2. `uv run pytest` completo (hoy 106 en verde).
3. Humo en CPU de `predecir`, de `ajustar` y de un tramo corto de `entrenar` con las opciones nuevas.
4. En el cluster: el primer reajuste de cada familia se revisa (métrica de validación sin NaN, que existan las predicciones) antes de encolar el resto. Una pasada de `predecir` sin control debe reproducir la AUL de `predicciones_test.parquet` de la corrida original (coherencia, como en la fase 7).
5. Los informes se escriben con las tablas generadas por `evaluar`, y los números del texto se contrastan con ellas antes de entregar.
