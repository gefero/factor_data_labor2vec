# Subplanes: fase 0 (entorno) y fase 1 (datos)

Estado: borrador para aprobación (2026-10-15). Complementa [PLAN.md](PLAN.md), secciones 6 y 8.

## Hechos nuevos que condicionan estas fases

- **Faltantes codificados como 0**, no como nulos ("Sin definir" según los descriptores). Vale para:

  | Variable | Filas con 0 |
  |---|---|
  | `letra`, `r32`, `r34` | 20.838 (las mismas filas en las tres) |
  | `tramo_empleo` | 5.206 |
  | `provi` | 3.857 |
  | `sexo` | 57.948 |

  `fnacim` sí usa nulo.
- **Jerarquía exacta**: `r32 == r34 // 100` en el 100 % de las filas.
- **Exclusión acordada** (sin `sexo` o sin `fnacim`): 20.949 trabajadores (3,5 %). Casi todos los que no tienen sexo tampoco tienen año de nacimiento.
- **Descriptores descargados** en `data/docs/Anexo_estadistico_y_descriptores.xlsx` (sitio del MTEySS). Traen:
  - las tablas de códigos (`DESC_SEXO`, `DESC_PROVI`, `DESC_LETRA`, `DESC_R32`, `DESC_R34`, `DESC_TRAMO EMPLEO`, `DESC_TR_ANIO_INI`). El código `provi = 1` es "Partidos de GBA" y `2` es CABA: el GBA va separado de la provincia de Buenos Aires;
  - **cuadros de validación muestra contra población**: empleo mensual (C1.1), por sexo, rama, edad, provincia y tamaño (C1.2-C1.8), y remuneración media y percentiles (C2.1-C2.5). Sirven para verificar que leemos los datos igual que el MTEySS.
- **Python local roto**: el Python 3.14 del sistema no tiene el módulo `decimal`. Se usa un Python gestionado por `uv`.
- **CCAD operativo**: la SSH funciona desde el contenedor (`grosati@mendieta`, a través del agente). En el cluster ya están `uv` y Python 3.13, y la receta probada de `land2vec` (torch 2.11.0 `cu126`, entorno empaquetado en `.tar` y descomprimido en `/scratch`).
- **Licencias**: los dos repos de referencia son MIT. Se puede copiar código citando la fuente.
- **Disco local**: quedan ~5 GB libres. `data/mler_csv.rar` (388 MB) duplica al CSV.

---

## Fase 0: entorno, estructura y prueba en GPU

**Objetivo**: un proyecto instalable que corra igual en el contenedor (CPU, desarrollo) y en Mendieta (GPU), con el backbone de life2vec andando en las dos máquinas.

### 0.1 Repositorio

- `git init`, con un `.gitignore` que excluya `data/`, `*.parquet`, `*.rar`, `modelos/`, `logs/` y `.venv/`. Los datos nunca entran al repo.
- Estructura:

```
pyproject.toml
src/labor2vec/
  datos/          # fase 1: ingesta, descriptores, QA, población, splits
  tokens/         # fase 2
  modelo/         # backbone copiado de life2vec-light (MIT, con atribución)
  tareas/         # fases 4 y 6: MLM+SOP, T1
  utiles/         # semillas, rutas, configuración, logging
configs/          # YAML por experimento
scripts/cluster/  # adaptados de land2vec: entorno, empaquetado, sincronización, sbatch
tests/
informes/         # salidas de QA y reportes (HTML/MD)
plans/            # planes (este archivo y PLAN.md)
data/             # MLER.csv y docs/ quedan donde están; se agrega processed/
```

- Configuración con YAML y dataclasses, sin Hydra. Hydra suma complejidad y el repo completo la usa sobre todo para combinar los configs daneses.
- Logging con el CSV logger de Lightning más TensorBoard.

### 0.2 Python y dependencias

- **Python 3.13** gestionado por `uv` (`--python-preference only-managed`), igual en local y en el cluster (variante x86-64 base, porque los nodos no tienen AVX2).
- Grupos de dependencias en `pyproject.toml`:

  | Grupo | Paquetes | Dónde |
  |---|---|---|
  | `datos` | duckdb, polars, pyarrow, numpy, openpyxl | local y cluster |
  | `modelo` | torch 2.11.0, lightning, torchmetrics, performer-pytorch | local con CPU, cluster con `cu126` |
  | `analisis` | scikit-learn, umap-learn, pacmap, matplotlib | más adelante |
  | `dev` | pytest, ruff | local |

- torch se instala desde el índice correcto según la máquina: CPU local (~200 MB) o `cu126` en el cluster. No se instala CUDA en el contenedor, por el espacio en disco.
- **Riesgo a validar**: que `performer-pytorch` funcione con torch 2.11 y Python 3.13. El código de referencia usa torch 2.3 y Python 3.11. Si falla, primero se prueba con su versión más nueva; si sigue fallando, se copia el módulo FAVOR+ (son ~80 líneas en `performer.py`).

### 0.3 Backbone

- Copiar de `life2vec-light` los archivos `transformer/{embeddings,attention,performer,modules,transformer,transformer_utils}.py` y `models/pretrain.py` a `src/labor2vec/modelo/`, con un encabezado de atribución.
- Ajustar imports y quitar el `vocab_size` hardcodeado.
- No se cambia nada de la lógica en esta fase.
- **Prueba de humo (CPU, en `tests/`)**: con una configuración mínima (hidden 32, 2 capas, vocabulario 50, largo 128), una pasada hacia adelante y hacia atrás de MLM+SOP con lotes aleatorios. La pérdida tiene que bajar en 20 pasos sobre un lote fijo (sobreajuste intencional).

### 0.4 Cluster (Mendieta), según el skill `ccad-unc`

**Reglas que aplican** (skill: reglas y "Seguridad para agentes"):
- Todo cálculo va por Slurm. En la cabecera sólo se sincroniza, se instala y se encola.
- Mendieta es sólo para cómputo con GPU, con 10 cores por GPU: el job mínimo es `--cpus-per-task=10 --gres=gpu:1`.
- Slurm cobra lo reservado, no lo usado: pedir el tiempo justo y revisar `seff` al terminar.
- Conexión sólo con el agente SSH del contenedor (`grosati@mendieta`): cada `ssh` ejecuta un comando y se cierra. La clave privada no se lee ni se copia.
- **Confirmación previa** para encolar jobs, cancelarlos, borrar o sobrescribir archivos en el cluster e instalar fuera del proyecto. Consultar estado y leer logs no la necesita.

**Pasos**:
1. **Código** (`scripts/cluster/sincronizar_mendieta.sh subir`):
   - rsync del árbol de trabajo a `~/labor2vec` (no un `git clone`), sin datos;
   - crea `logs/` antes de cualquier `sbatch`, porque Slurm no crea el directorio de `--output` y el job falla sin aviso.
2. **Entorno** (`entorno_mendieta.sh`, en la **cabecera**, porque los nodos no tienen internet):
   - Python 3.13 de `uv` en la variante x86-64 base (los nodos Ivy Bridge no tienen AVX2), con `--python-preference only-managed`;
   - `uv sync --frozen --extra cu126` (A30 = sm_80);
   - todo dentro de `~/labor2vec`, sin instalar nada fuera del proyecto.
3. **Empaquetado** (`empaquetar_entorno.sh`, lecciones §1):
   - el CPython de uv más el `site-packages` del venv en un solo `entorno/py313_labor2vec.tar`;
   - cada job lo descomprime en el `/scratch` del nodo con `entorno_scratch.sh`, porque correrlo desde el `/home` (NFS) hace que `import torch` tarde minutos;
   - hay que rehacer el `.tar` cada vez que cambie el venv.
4. **Job de prueba** (`prueba_mendieta.sbatch`), en `short`, 30 min, 1 A30, 10 cores. Sigue la plantilla del skill:
   - `. /etc/profile` antes de `set -euo pipefail` (lecciones §3);
   - `python3 -u -X faulthandler`, la hora de cada etapa y el código de salida en el log (lecciones §2);
   - lanza Python igual que lo hará el job real (lecciones §2): mismo entorno desde `/scratch` y mismo script;
   - nombre propio (`l2v_prueba`), para no chocar con otros jobs pendientes (lecciones §4).

   Qué mide:
   - el tiempo de copia del entorno y de `import torch`;
   - la GPU detectada y un matmul;
   - el backbone con la configuración del paper (8,5 M parámetros, 2560 tokens), con lotes de 4 a 32 en fp32 y bf16: pasos/s, secuencias/s y memoria pico.

   Esos números permiten estimar el costo de la fase 4.
5. **Antes de encolar**:
   - `sbatch --test-only` para estimar el arranque (no encola);
   - **te pido confirmación**.
6. **Después**:
   - leer el log y revisar `sacct` y `seff`;
   - traer el log con `sincronizar_mendieta.sh bajar`;
   - volcar las mediciones en `informes/fase0.md`.

**Para la fase 4** (no aplica a esta prueba): el pre-entrenamiento va a superar 1 h, así que hace falta lo que pide el skill para trabajos largos:
- checkpoints atómicos y retomar con el mismo comando;
- parar solo antes del límite con un plazo absoluto (`--hasta-epoch`, lecciones §9 y §12);
- encadenar jobs con `afterany`;
- probar la **retoma en GPU**, no sólo el arranque (lecciones §5, los estados RNG deben volver a la CPU);
- hacer una prueba de 1 h del trabajo real en `short` antes de lanzar en `multi`.

Al publicar, citar el uso de recursos del CCAD.

### Entregables y criterio de aceptación de la fase 0

1. `uv sync` instala sin errores en local, y `pytest` pasa (incluida la prueba de humo en CPU).
2. El job de prueba en Mendieta termina `COMPLETED`, y el log muestra el tiempo de import, la GPU detectada, la memoria pico y los pasos por segundo del modelo de tamaño del paper.
3. Una nota en `informes/fase0.md` con las versiones exactas y las mediciones.

---

## Fase 1: ingesta, validación y base de población

**Objetivo**: datos limpios, tipados y validados contra las cifras oficiales; tablas de trabajadores y relaciones; población elegible y splits fijos. Todo con scripts idempotentes: se pueden correr de nuevo y dan el mismo resultado.

### 1.1 Ingesta

Script `labor2vec.datos.ingesta`:

- Lee `MLER.csv` con DuckDB y tipos explícitos.
- Registra el tamaño y el hash del CSV en un archivo de metadatos.
- Ordena por `(id_trabajador, tiempo, id_relacion)` y guarda `data/processed/panel.parquet` (zstd, ~220 MB).
- Columnas derivadas:
  - `anio`;
  - `mes_idx` (0-311);
  - `fecha` (día 15 del mes);
  - `dias_desde_1996` (insumo de la posición absoluta).
- `pondera` se descarta: es constante.
- Los ceros de código se recodifican a nulo con un flag por variable, para no mezclar "dato 0" con "sin dato".

### 1.2 Descriptores

Script `labor2vec.datos.descriptores`:

- Extrae las hojas `DESC_*` a `data/processed/descriptores/*.parquet`.
- Resuelve las fórmulas de `DESC_LETRA` (`=+A5+1`) para obtener los códigos numéricos.
- Agrega una tabla **propia y documentada** de regiones para las provincias (GBA, CABA, Pampeana, NOA, NEA, Cuyo, Patagonia). Sirve para validar el espacio de conceptos en la fase 5.
- Verifica que todos los códigos presentes en el panel tengan descripción, y que se cumpla la jerarquía `r34 → r32 → letra`.

### 1.3 Validación contra los cuadros oficiales

Script `labor2vec.datos.validacion`:

- Recalcula desde el panel las columnas "Muestra" de C1.1 (empleo mensual), C1.2 (por sexo), C1.3 (por letra), C1.5 (por provincia), C1.6 (por tamaño), C2.1 (remuneración media mensual) y C2.2 (percentiles).
- **Criterio**: coincidencia exacta en los recuentos y diferencias < 0,5 % en remuneraciones (la muestra publicada podría ser previa al ruido de anonimización).
- Si no coincide, primero se busca la definición que usó el MTEySS (por ejemplo, si "empleo" cuenta relaciones con `rem_tot > 0` o filas) y se documenta.
- Es la prueba de que interpretamos bien cada variable antes de construir nada encima.

### 1.4 Reporte de calidad

Script `labor2vec.datos.qa` → `informes/qa_mler.html`. Reproduce y amplía la sección 2.3 de PLAN.md:

- faltantes por variable (códigos 0 y nulos) y por año;
- remuneraciones: ceros, valores < 30, > 10 M y zona microagregada;
- largos por trabajador;
- brechas dentro de las relaciones;
- meses con varias relaciones;
- censura izquierda (1996-01) y derecha (2021-12);
- edades implícitas fuera de rango;
- concentración de `r34` (categorías raras que irían a `[UNK]` en la fase 2).

### 1.5 Tablas derivadas

- **`trabajadores.parquet`**, una fila por trabajador:
  - `sexo`, `fnacim`;
  - primer y último mes, meses con registro, número de relaciones;
  - `excluido` y `motivo_exclusion` (sin sexo o sin `fnacim`; 20.949 casos esperados).
- **`relaciones.parquet`**, una fila por (trabajador, relación):
  - atributos estáticos (`provi`, `letra`, `r32`, `r34`, `tr_anio_ini`);
  - inicio, fin, meses, número de brechas;
  - `censura_izq` y `censura_der`.

### 1.6 Splits

- Asignación fija 70/15/15 por `id_trabajador` entre los no excluidos, con semilla fija, guardada en `data/processed/splits.parquet`. Es el split global del paper: las tareas de fine-tuning lo heredan, filtrando su cohorte.
- Se hace en la fase 1 porque la fase 2 tiene que ajustar los cortes de percentil salarial **sólo con train**, como hace el paper con sus bins.

### 1.7 Salario relativo

- Los cortes de percentil por mes, calculados sobre train y con `rem_tot > 0`, se guardan en `data/processed/cortes_salario_mensual.parquet` (312 meses × 99 cortes).
- Al panel se le agrega una columna `pct_rem_mes` (0-100), con `rem_tot = 0` marcado aparte.
- La cantidad de bins y su conversión a tokens se deciden en la fase 2. Acá queda sólo el insumo continuo.

### 1.8 Pruebas y cluster

- Pruebas en `tests/` sobre el panel procesado:
  - unicidad de `(trabajador, relación, mes)`;
  - atributos constantes dentro de cada relación;
  - `sexo` y `fnacim` constantes dentro de cada trabajador;
  - jerarquía de ramas;
  - splits disjuntos que cubren a todos los elegibles.
- Sincronizar a `~/labor2vec/data/processed` en Mendieta sólo los parquet procesados (~300 MB), no el CSV.

### Entregables y criterio de aceptación de la fase 1

1. `data/processed/` con `panel`, `trabajadores`, `relaciones`, `splits`, `cortes_salario_mensual` y `descriptores/`. Cada script se corre con un comando, sin pasos manuales.
2. La validación contra C1.x y C2.x pasa con los criterios de 1.3, o los desvíos quedan documentados con su explicación.
3. `informes/qa_mler.html` revisado por vos.
4. `pytest` pasa.

---

## Puntos para confirmar antes de arrancar

1. **`git init`** del proyecto con la estructura de 0.1. ¿Tenés un remoto (GitHub/GitLab) al que haya que subirlo, o queda local por ahora?
2. **`data/mler_csv.rar`** (388 MB) duplica al CSV, y hay poco disco. ¿Lo borro, o lo dejo?
3. **Job de prueba en Mendieta** (fase 0.4): 1 h como máximo en `short`, 1 GPU. Te aviso antes de encolarlo, salvo que me autorices de antemano para los jobs de prueba en `short`.
