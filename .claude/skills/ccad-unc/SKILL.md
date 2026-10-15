---
name: ccad-unc
description: Usar los clusters de UNC Supercómputo / CCAD (Universidad Nacional de Córdoba) - Mendieta (GPU A30), Serafín, Eulogia, Mulatona - por SSH y Slurm. Usar cuando haya que conectarse a *.ccad.unc.edu.ar, preparar un entorno de Python/PyTorch, escribir o encolar scripts sbatch, elegir partición, mover archivos con rsync, seguir jobs (squeue, sacct, seff, jobmetrics) o traer resultados. También ante dudas de normas de uso del CCAD, colas, límites de tiempo o errores de jobs (TIMEOUT, DependencyNeverSatisfied, imports lentos).
---

# UNC Supercómputo (CCAD-UNC)

Fuente: la wiki oficial, https://wiki.ccad.unc.edu.ar/ (36 páginas, fuente en https://gitlab.com/ccad/wiki-rtd).
Resumen página por página, con URLs: [references/wiki.md](references/wiki.md).
Lo aprendido en la práctica, que no está en la wiki: [references/lecciones-mendieta.md](references/lecciones-mendieta.md). **Leelo antes del primer job.**
Plantilla mínima de job de GPU: [references/plantilla-sbatch.sh](references/plantilla-sbatch.sh).
Opcional, sólo si estás en el repo land2vec: `scripts/cluster/` tiene una implementación completa que ya funcionó (tuneo de un autoencoder en Mendieta, con empaquetado del entorno, cadenas de jobs y reanudación). Fuera de ese repo no existe; las lecciones y la plantilla alcanzan.

## Reglas (de la wiki y del CCAD)

1. **Todo cálculo va por Slurm.** En la cabecera (nodo de login) sólo se edita, instala, compila y encola. Nada de procesos pesados ahí.
2. **Mendieta es sólo para aplicaciones que usan GPU, y hay que pedir 10 cores por cada GPU** (aviso al iniciar sesión). El job mínimo es medio nodo: `--cpus-per-task=10 --gres=gpu:1`.
3. **Respetar los límites de la partición** (tabla abajo). Partir un trabajo largo en jobs más cortos que guardan y se retoman es una práctica que la propia wiki muestra (páginas *Slurm signals* y *Quantum Espresso*).
4. **Slurm cobra lo reservado, no lo usado.** Revisar la eficiencia con `seff`. El CCAD tiene tarifario por hora-core (RR-469-2021).
5. **Las claves SSH son personales.** La clave privada nunca se copia, se muestra ni se pide. Ver "Seguridad para agentes".
6. **Citar el uso de recursos** en publicaciones: https://supercomputo.unc.edu.ar/equipamiento/citar-recursos/

## Clusters

| | Mendieta | Mulatona | Eulogia | Serafín | Boogie |
|---|---|---|---|---|---|
| Uso | Cómputo con GPU | Trabajos pequeños | Sólo software preinstalado | Propósito general | JupyterHub |
| Nodos | 19 | 7 | 32 | 64 | 8 |
| Cores/nodo | 20 (Xeon E5-2680 v2, **Ivy Bridge, sin AVX2**) | 32 (Xeon E5-2683 v4) | 64C/256T (Xeon Phi 7210) | 64 (EPYC 7532) | 128 (EPYC 7B12) |
| GPU/nodo | **2 × NVIDIA A30 (24 GB)** | - | - | - | - |
| RAM/nodo | 64 GB | 128 GB | 96 GB | 256 GB | 512 GB |
| /scratch | 400 GB SSD | 95 GB SSD | 200 GB SSD | 1 TB NVMe | - |
| Job mínimo | ½ nodo (10 cores, 1 GPU) | 1 core | 1 nodo | 1 nodo | 1 core |
| Host | mendieta.ccad.unc.edu.ar | mulatona… | eulogia… | serafin… | (JupyterHub) |

## Particiones

| | Mendieta | Mulatona | Eulogia | Serafín |
|---|---|---|---|---|
| `short` | hasta 1 h (default) | hasta 1 h, 1 nodo (default) | hasta 1 h | hasta 1 h (default) |
| `multi` | hasta 2 días | - | hasta 4 días (default) | hasta 2 días |
| `mono` | - | hasta 2 días, 1 nodo | - | - |

`short` suele arrancar mucho antes que `multi`. Ver la cola real con `sinfo` y `squeue -p multi`.

## Almacenamiento

- **`/home`**: remoto (NFS), permanente y compartido por la cabecera y los nodos. **Lento para mucha E/S**, incluido cargar bibliotecas grandes (un `import torch` desde el /home tardó 710 s en un nodo).
- **`/scratch`**: disco local de cada nodo, rápido. **Se borra al terminar el job.** Para conservar algo: copiarlo al /home al final o usar `sgather`.
- **`/tmp`**: vive en la RAM del nodo (hasta 50 %). Sólo archivos chicos.

## Flujo de trabajo

### 0. Conexión
```bash
ssh -o BatchMode=yes USUARIO@mendieta.ccad.unc.edu.ar 'comando'
```
Sólo claves SSH. Con BatchMode, si la clave no está disponible falla en vez de pedir contraseña. La primera conexión puede necesitar `-o StrictHostKeyChecking=accept-new`. Para ver si los clusters están en línea: https://stats.uptimerobot.com/eLhTV5CMni

### 1. Llevar el código y los datos
```bash
rsync -avz --exclude __pycache__ src scripts setup.py USUARIO@mendieta.ccad.unc.edu.ar:proyecto/
```
Copiar el árbol de trabajo, no un `git clone`, si hay cambios sin commitear o archivos LFS (el cluster no tiene git-lfs). **Crear `logs/` antes de encolar:** Slurm no crea el directorio de `--output` y el job falla sin aviso.

### 2. Entorno de Python
- La wiki recomienda un entorno propio en el home (Miniconda). También funciona `uv` (un solo binario en `~/.local/bin`).
- **Se instala en la cabecera**, porque los nodos de cómputo no tienen internet.
- En Mendieta: torch con el canal `cu126` o `cu128` (A30 = sm_80; driver 580 visto en 2026-10). Python **x86-64 base**, no v3: los nodos no tienen AVX2.
- **No correr el entorno desde el /home:** empaquetarlo en un `.tar` y descomprimirlo en el `/scratch` al empezar cada job (36 s para 6,6 GB). Ver lecciones §1 y `scripts/cluster/empaquetar_entorno.sh` y `entorno_scratch.sh`.

### 3. Probar antes de lanzar lo largo
Un job corto en `short` que haga lo mismo que el job real: el mismo entorno, la misma forma de lanzar Python, unos pasos de cálculo en la GPU, con `python -u` y la hora de cada etapa. Ver `scripts/cluster/prueba_mendieta.sbatch`. Mejor aún, **una hora del trabajo real en `short`**: así aparecieron los dos errores que una prueba sintética no mostró (lecciones §3 y §5).

### 4. Script sbatch (plantilla para Mendieta)
```bash
#!/bin/bash
#SBATCH --job-name=nombre
#SBATCH --partition=multi            # short: hasta 1 h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10           # 10 por GPU (regla de Mendieta)
#SBATCH --gres=gpu:1                 # GPUs por nodo; 2 = nodo completo con 20 cores
#SBATCH --time=2-00:00               # días-horas:minutos
#SBATCH --output=logs/%x-%j.out      # logs/ tiene que existir
#SBATCH --mail-type=BEGIN,END,FAIL   # opcional
#SBATCH --mail-user=correo@dominio

. /etc/profile                       # ANTES de `set -u` (lecciones §3)
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# entorno desde /scratch, ver lecciones §1
python3 -u programa.py
```
- Las líneas `#SBATCH` se leen hasta el primer comando: después se ignoran.
- Con `--gres`, `CUDA_VISIBLE_DEVICES` numera desde 0 las GPUs asignadas.
- **Slurm guarda una copia del script al encolar:** si lo corregís, hay que cancelar y volver a encolar.

### 5. Encolar y seguir
```bash
sbatch job.sh                                    # devuelve el JOBID (--parsable: sólo el número)
sbatch --test-only job.sh                        # estima el arranque sin encolar
squeue --me --start                              # cola y arranque estimado
sbatch --dependency=afterok:JOBID job2.sh        # encadenar
scontrol update JobId=N MailType=END,FAIL MailUser=x@y   # cambiar un job pendiente
scontrol show job N; scancel N
sacct -X -j N -o JobID,State,Start,Elapsed,ExitCode
seff N; jobmetrics N                             # eficiencia; Grafana (stats.ccad.unc.edu.ar)
srun --gres=gpu:1 --pty bash                     # sesión interactiva con GPU (sólo Mendieta)
```
Motivos de espera comunes: `Priority`, `Resources`, `Dependency`, `DependencyNeverSatisfied` (la dependencia falló: cancelar y reencolar) y `AssocMaxJobsLimit`.

### 6. Trabajos que superan el límite de tiempo
- El programa tiene que **guardar checkpoints y poder retomar con el mismo comando**.
- Hacerlo **parar solo antes del límite**, calculando el tiempo restante con `squeue -h -j $SLURM_JOB_ID -o %e` menos un margen (5 min en `short`, 15 en `multi`).
- Como refuerzo, `#SBATCH --signal=B:TERM@300` manda una señal al script 300 s antes del límite.
- Volver a encolar el mismo script cuantas veces haga falta.
- Para que dos jobs no pisen el mismo trabajo: `--dependency=singleton` (lecciones §4) o `afterany`.

### 7. Traer resultados
```bash
rsync -avz --exclude ckpt.pt USUARIO@mendieta.ccad.unc.edu.ar:proyecto/resultados/ resultados/
```

## Seguridad para agentes

- **Autenticación:** usar sólo el agente SSH del usuario (`SSH_AUTH_SOCK`, que VS Code reenvía a los devcontainers). **Nunca** pedir, leer, copiar ni pegar la clave privada. Si el usuario ofrece pasarla, rechazarlo y proponer el agente o una clave nueva cuya pública agregue él (procedimiento "Desde otra computadora" de la wiki).
- **Conexiones:** cada `ssh` ejecuta un comando y se cierra; no dejar sesiones ni ControlMaster abiertos.
- **Confirmar con el usuario antes de:**
  - encolar jobs largos (consumen horas de su cuenta);
  - cancelar jobs;
  - borrar o sobrescribir archivos en el cluster;
  - instalar cosas fuera del proyecto.
  Consultar el estado y leer logs no necesita confirmación.
- **Banner:** las conexiones muestran un recuadro de bienvenida. Se puede filtrar del output, pero **leerlo primero**: ahí aparecen avisos de normas (por ejemplo, "10 cores por GPU"). El aviso de *Novedades* sólo sale en sesiones interactivas.
- **Aviso "post-quantum" de OpenSSH ≥ 10:** el servidor usa OpenSSH 8.0. No impide trabajar.
- **Antes de commitear o subir cualquier cosa,** buscar claves o tokens en lo que se va a publicar.

## Ayuda

- Wiki con buscador: https://wiki.ccad.unc.edu.ar/
- Zulip de usuarios: https://ccadunc.zulipchat.com/
- Soporte: `soporte@ccad.unc.edu.ar`, indicando usuario, cluster, directorio, script sbatch y archivo con el error.
- Reunión virtual: https://calendly.com/ccad-soporte-ale/consulta
- Pedido de cuenta: https://forms.gle/xu4owamaRKqpxWoy7 (hay que repetirlo en cada reempadronamiento).
- Cómputo intensivo con exclusividad: "Ventanilla permanente de uso intensivo" en supercomputo.unc.edu.ar/servicios.
