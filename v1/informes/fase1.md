# Fase 1 de v1: codificación con grilla mensual y huecos explícitos

Generado el 2026-11-02. Plan: `plans/PLAN_v1.md`. Código en `v1/src/labor2vec/tokens/`, pruebas en `v1/tests/test_grilla.py` (21 nuevas), cifras con `v1/scripts/largos_grilla.py` (resultado completo en `informes/largos_grilla.json`).

## Qué se hizo

- **Vocabulario** (`tokens/vocabulario.py`): dos tokens nuevos, `SINREG_INT` y `SINREG_SAL`, de categoría `sinreg`, al final del vocabulario (ids 519 y 520; 521 en total). **No se guardan** en `vocabulario.parquet`: ese archivo es el de v0, compartido por el enlace `data/processed`. `cargar` y `ajustar` los agregan, y una prueba verifica que los ids de v0 no se mueven.
- **Eventos** (`tokens/documento.py`): `Eventos` gana el campo `tipo` (0 empleo, 1 intermitencia, 2 salida; por defecto todos de empleo, así que el código heredado no cambia).
- **Grilla**: `con_grilla(ev, fin)` agrega un evento por cada mes sin registro entre el primer mes observado y el fin de la ventana. El tipo se define sólo con lo que hay hasta ese fin: intermitencia si hay un registro posterior, salida si el último registro quedó atrás. Antes del primer registro no se agrega nada.
- **Bloque**: un mes sin registro es `SINREG_x [SEP]` (2 tokens); la fecha, la edad y el ciclo de segmentos siguen la regla de los eventos de empleo.
- **Opciones**: `Opciones.grilla` (por defecto `False`, o sea, exactamente v0) y `Opciones.sinreg` (`"dos"` | `"uno"`, el control de la fase 5). Todas las configuraciones de `v1/configs/` traen `grilla: true`.
- **Intervenciones de la fase 9**: la grilla se arma después de la intervención y termina donde termina la historia intervenida (`fin_de_la_historia`): con `sin_ultimo_anio` termina 12 meses antes y con `atrasar_2_anios`, 24. Sin esto, borrar el último año habría fabricado meses de salida.
- **`DatasetPretrain.largos()`** cuenta los meses sin registro de forma vectorizada y coincide **exactamente** con el largo del documento armado (probado con y sin ventana, con `sin_ultimo_valor` y con eventos compactos, y verificado además en 5.000 trabajadores reales).
- El `.venv` y los datos son los de v0 (ver el plan, sección 5). Las pruebas de v0 siguen valiendo: 181 pasan y 6 se saltan.

## Cifras sobre el corpus real (train, ventana a 2015-12, 344.025 trabajadores)

Tokens de eventos por persona (sin los 4 del background y antes de la augmentación):

| | v0 | v1 (grilla) |
|---|---|---|
| Mediana | 264 | **508** (1,92 veces) |
| Media | 426 | 589 (1,38 veces) |
| p90 | 1.152 | 1.228 |
| p99 | 1.452 | 1.462 |
| Truncados a 2.560 | 0,119 % | 0,124 % |

El largo crece mucho en quien tiene historia corta y casi nada en quien tiene historia larga. La mediana se duplica; la cola, que es la que fija el costo del lote, apenas cambia. El plan estimaba 2,5 veces con la mediana de v0; resultó 1,9. El costo de cómputo sigue la media (1,4 veces), no la mediana.

**Meses de la grilla** (primer mes observado hasta 2015-12): mediana 154 meses por persona, de los cuales una mediana de 63 son sin registro.

| | Proporción de los meses de la grilla |
|---|---|
| Con empleo | 45,8 % |
| `SINREG_INT` (intermitencia) | 17,0 % |
| `SINREG_SAL` (salida) | 37,2 % |

- El 62,5 % de las personas termina con salida en 2015-12; el 58,7 % tiene al menos un mes de intermitencia.
- Los tokens `SINREG` son el 27,7 % de los tokens de eventos de v1.
- El 19,9 % de las personas empieza en enero de 1996 (primer mes de la MLER): tienen censura a la izquierda, no se puede saber si trabajaban antes.

**Cohorte de T1** (85.852 personas en train; los 12 meses de 2015 registrados):

| | v0 | v1 |
|---|---|---|
| Mediana de tokens de eventos | 732 | 802 |
| p99 | 1.764 | 1.782 |
| Meses `SINREG` | | 17,2 % de la grilla |
| Con algún mes de intermitencia | | 65,2 % |
| Terminan en salida | | 0 % (por construcción) |

Para T1 el costo de v1 es chico (+10 % en la mediana) y el contenido nuevo es la intermitencia de la historia previa, no la salida.

## Cambios posteriores (2026-11-02 y 03)

- **SOP:** con la grilla, la permutación libre de bloques de 6 y de 2 tokens movía los `[SEP]` en el 86 % de los documentos perturbados (0 % en v0): el AUC de SOP de 0,875 en el humo con grilla (contra 0,56 del control sin grilla) era un atajo. Ahora `_orden_sop` intercambia sólo bloques del mismo largo; los `[SEP]` movidos pasan a 0 % y con un solo largo el resultado es idéntico al de v0 (test).
- **Nombres:** `ANT_k` pasa a `EMPINI_k` (inicio de actividades de la empresa, `tr_anio_ini`). Ver la tabla de variables en `plans/PLAN_v1.md`, sección 3. La rama sigue siendo una sola variable (r34, con r32 para las 19 ramas raras); la letra no entra.
- **Humo en CPU** (15 épocas, muestra del 1 %, mismos hiperparámetros): control sin grilla (= v0) perplejidad 32,0 contra 78,7 de la unigrama (v0 dio 32,3); con grilla 19,4 contra 61,0. Los `SINREG` son el 19,4 % de los objetivos del MLM y el 2,7 % de la pérdida (`SINREG_SAL`: NLL 0,28, acierto 0,99; `SINREG_INT`: 0,72 y 0,92), así que la perplejidad de v1 no se compara con la de v0. Script: `scripts/mlm_por_categoria.py`.

## Lo que esto cambia o confirma del plan

- **Costo del pre-entrenamiento**: unas 1,4 veces los tokens de v0 (no 2,5). Estimación para la fase 3: 7-10 h en lugar de 8-15 (se mide en la fase 2).
- **MLM**: el 27,7 % de los tokens son `SINREG`. Una vez que empieza la salida todos los meses siguientes son `SINREG_SAL`, así que enmascarar uno es casi trivial por el contexto. Habrá que mirar en la fase 2 cuánto diluyen la pérdida y, si hace falta, bajar su probabilidad de enmascarado.
- **Pre-entrenamiento y T1 ven poblaciones distintas**: la salida es el 37 % de los meses en el pre-entrenamiento y no aparece nunca en la cohorte de T1.
- **Análisis** (`analisis/espacio.py`, `personas.py`): construyen `Opciones` a mano o suponen eventos de 6 tokens y no consideran la grilla. Se adaptan en la fase 6, no antes de tener un modelo.
- Los scripts de `v1/scripts/cluster/` y las rutas `~/labor2vec/...` de las configuraciones siguen apuntando a v0; se arreglan en la fase 2.

## Criterios de aceptación de la fase 1

1. Tests en verde: ✔ (181 pasan, 6 se saltan por falta de `node`, de `panel.parquet` y de los vectores de la fase 7).
2. Informe de largos y de proporción de meses `SINREG` por tipo: ✔ (arriba).
3. `largos()` exacto frente al documento armado: ✔.
