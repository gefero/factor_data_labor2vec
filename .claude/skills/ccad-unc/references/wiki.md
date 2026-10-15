# Wiki de UNC Supercómputo: resumen por página

Base: https://wiki.ccad.unc.edu.ar/ (fuente Markdown de cada página en `<url>/_sources/<ruta>.md`; repo https://gitlab.com/ccad/wiki-rtd).
Leída completa el 2026-10-09 y el 2026-10-10, sin cambios entre ambas. Ante una duda puntual, releer la página original: puede haber cambiado.

## Primeros pasos

### Desde cero: `empezar/desde-cero.html`
Para quien nunca usó un cluster: la presentación "Desde 0 al CCAD" (con video) y *The Missing Semester* del MIT (https://missing.csail.mit.edu/, traducción parcial en https://missing-semester-esp.github.io/).

### Guía de inicio: `empezar/abrir-cuenta.html`
1. **Requisitos:** un par de claves SSH y completar el formulario https://forms.gle/xu4owamaRKqpxWoy7
2. **Acceso:** llega un mail con el usuario. A los clusters se entra **sólo con clave SSH**. Los servicios web (JupyterHub, chat) usan una contraseña aparte.
3. **Software:** módulos preinstalados o Spack en el home. **Es obligatorio usar Slurm; no correr procesos pesados en los nodos de login.** Monitorear la eficiencia de los jobs.
4. **Publicaciones:** citar el uso de recursos (https://supercomputo.unc.edu.ar/equipamiento/citar-recursos/).

### Acceso: `empezar/acceso/index.html`
- **Clusters:** Serafín, Mendieta, Eulogia y Mulatona.
- **JupyterHub:** https://jupyterhub.ccad.unc.edu.ar (usuario y contraseña).
- **LLMs:** https://chat.ccad.unc.edu.ar, con Open WebUI y Ollama. Tiene API: crear una API key y usar `https://chat.ccad.unc.edu.ar/` como URL base, por ejemplo `curl -H "Authorization: Bearer sk-…" https://chat.ccad.unc.edu.ar/api/models`.

Subpáginas:
- **Claves (`concepts.html`):** la pública es la cerradura y la privada la llave. La privada nunca se comunica. Ante robo o pérdida, avisar a soporte. La seguridad del cluster depende del uso responsable de la autenticación.
- **Generar claves (`generate-keys.html`):** `ssh-keygen -t ed25519`, aceptando los valores por defecto. La pública es `~/.ssh/id_ed25519.pub`. En Windows, MobaXterm instalable (la versión portable da problemas con las claves).
- **Conexión (`connect.html`):** `ssh $USUARIO@{serafin,mendieta,eulogia,mulatona}.ccad.unc.edu.ar`. Se entra al **nodo cabecera**, que no es para calcular: sólo prepara y envía trabajos. No se pide contraseña.
- **Problemas de conexión (`problems.html`):**
  - Ver el estado de los clusters: https://stats.uptimerobot.com/eLhTV5CMni
  - Verificar que la privada corresponda a la pública: `ssh-keygen -y -f ~/.ssh/ed25519`.
  - Permisos: `chmod 600` a la clave y `chmod 700` a `~/.ssh`.
  - Forzar la clave: `ssh -i …`; si no anda, `SSH_AUTH_SOCK=0 ssh -i …`. Para que quede: `ssh-add` o, en `~/.ssh/config`: `Host *.ccad.unc.edu.ar` / `User` / `IdentityFile` / `IdentitiesOnly yes`.
  - "Connection refused": puede ser el proveedor de internet, o un bloqueo temporal por varios intentos fallidos (esperar unas horas).
  - "This account is currently not available": la cuenta se dio de baja en el reempadronamiento; hay que llenar el formulario en cada reempadronamiento.
  - Si pide passphrase y no la recordás, generar claves nuevas y mandar la pública a soporte.
  - Para soporte, adjuntar la salida de `ssh -vvv …`.
- **Desde otra computadora (`new-machine.html`):** A, recomendado: generar un par nuevo en la máquina nueva, llevar la pública a la máquina habitual y desde ahí `ssh-copy-id -f -i id_laptop.pub $USUARIO@serafin.ccad.unc.edu.ar`. Las claves autorizadas son compartidas por todos los clusters. B, menos recomendado: copiar la privada original.
- **Enviar la clave a soporte (`send-keys.html`):** mandar el contenido de `id_ed25519.pub`.
- **Contraseña web (`reset-pass.html`):** `ssh -T $USUARIO@passwd.ccad.unc.edu.ar` genera una contraseña nueva, que se ve una sola vez. La contraseña es sólo para los servicios web; la clave SSH, para los clusters. MobaXterm puede cambiarla dos veces si tiene activo el navegador SSH.

### Ejecución de trabajos: `empezar/slurm.html`
- **Comandos:** `sbatch job.sh` (devuelve el JOBID), `squeue --me`, `sinfo`, `scancel JOBID`.
- **`squeue`:** columnas JOBID, PARTITION, NAME, USER, ST (PD pendiente, R corriendo, CG terminando), TIME, NODES, NODELIST(REASON).
  - Motivos de espera: `Resources`, `Priority`, `AssocMaxJobsLimit` (tope de jobs simultáneos), `Dependency`, `JobHeldUser`, `JobHeldAdmin`, `launch failed requeued held` y `Nodes required for job are DOWN, DRAINED or reserved`.
  - Listas completas: https://slurm.schedmd.com/job_state_codes.html y https://slurm.schedmd.com/job_reason_codes.html
- **Ejemplos:**
  - 1 core: `--nodes=1 --ntasks-per-node=1`.
  - Serafín, nodo completo: `--ntasks-per-node=64 --partition=short`.
  - Mendieta, 1 GPU: `--ntasks=1 --partition=short --gres=gpu:1`, y `nvidia-smi -L` para listar.
  - Eulogia, 2 nodos: `--ntasks-per-node=64 --cpus-per-task=4`.
  - Mulatona, 1 nodo: `--ntasks-per-node=32`.
  - Quantum Espresso y GROMACS: con `module load`; en Mendieta, `--cpus-per-task=10 --gpus-per-task=1 --gres=gpu:1 --partition=multi`.
- **Sesión interactiva:** `srun --pty bash`; con GPU (sólo Mendieta), `srun --gres=gpu:1 --pty bash`.
- **Útiles:**
  - `sbatch --test-only script.sh`: valida el script y estima el arranque sin encolar.
  - `sacct -X -S now-4weeks`: historial de jobs.
  - `sreport cluster AccountUtilizationByUser -t hours users=$USER start=… end=…`: uso de recursos.
  - `scontrol show job $JOBID`: verifica lo que Slurm entendió del preámbulo.

## Infraestructura

### Clusters disponibles: `infra/clusters.html`
Tabla comparativa:

| | Mendieta | Mulatona | Eulogia | Serafín | Boogie |
|---|---|---|---|---|---|
| Uso recomendado | Cómputo con GPU | Trabajos pequeños | Sólo software preinstalado | Propósito general | JupyterHub |
| Nodos | 19 | 7 | 32 | 64 | 8 |
| CPU | 20C/20T Xeon E5-2680 v2 | 32C/32T Xeon E5-2683 v4 | 64C/256T Xeon Phi 7210 | 64C/64T EPYC 7532 | 128C/128T EPYC 7B12 |
| GPU | 2 × A30 | - | - | - | - |
| RAM | 64 GB | 128 GB | 96 GB | 256 GB | 512 GB |
| /scratch | 400 GB SSD SATA | 95 GB SSD SATA | 200 GB SSD SATA | 1 TB SSD NVMe | - |
| Red | IB QDR 40 Gbps | IB QDR 40 Gbps | IB QDR 40 Gbps | IB HDR100 100 Gbps | Ethernet 25 Gbps |
| Trabajo mínimo | ½ nodo (10 cores, 1 GPU) | 1 core | 1 nodo | 1 nodo | 1 core |
| Compiladores | GCC 11, NVIDIA HPC SDK 22.3 | GCC 12, Intel 2021, OneAPI 2023 | GCC 12, Intel 2021 | GCC 12, AMD AOCC 4.1 | GCC 13 |
| MPI | Open MPI 4.1 | Open MPI 4.1 | Open MPI 4.1 | Open MPI 4.1 | - |
| Codename | ivb | bdw | knl | rome | bugi |

Al entrar a Mendieta aparece este aviso: "Este cluster ha sido actualizado con 2 GPUs NVIDIA A30 por nodo de cómputo y es para uso exclusivo de aplicaciones que utilizan GPUs. Por favor solicite 10 cores por cada GPU utilizada."

### Almacenamiento: `infra/almacenamiento.html`
- **`/home`:** sistema de archivos remoto, permanente y visible desde todos los nodos. Está optimizado para confiabilidad, no para E/S rápida; usarlo para guardar lo que se quiere conservar.
- **`/scratch`:** disco local de cada nodo, con XFS, rápido para E/S intensiva. **Se borra al terminar el job.** Para conservar archivos, poner `sgather /scratch/archivo prefijo` al final del script, que copia a `prefijo/<nodo>/archivo` (https://www.mankier.com/1/sgather).
- **`/tmp`:** está en la RAM del nodo, hasta el 50 %. Sólo archivos chicos; se borra al terminar si no quedan otros jobs del usuario en el nodo.

### Particiones de Slurm: `infra/slurm-particiones.html`

| | Mendieta | Mulatona | Eulogia | Serafín |
|---|---|---|---|---|
| short | hasta 1 h (default) | hasta 1 h, 1 nodo (default) | hasta 1 h | hasta 1 h (default) |
| multi | hasta 2 días | - | hasta 4 días (default) | hasta 2 días |
| mono | - | hasta 2 días, 1 nodo | - | - |

Si no se especifica, va a la partición por defecto. Se elige con `#SBATCH --partition=…`; el estado se ve con `sinfo`.

## Tutoriales

### Script de lanzamiento: `tutoriales/slurm-script.html`
- **Preámbulo:** `#!/bin/bash`; `--job-name`; opcional `--mail-type=ALL --mail-user=…`. Las líneas `#SBATCH` que aparecen después del primer comando se ignoran.
- **Recursos:**
  - `--partition`;
  - `--ntasks=P`, o `--nodes=N --ntasks-per-node=PPN`;
  - `--time=dd-hh:mm`, que conviene que sea cercano a la duración real porque al vencer el job se mata;
  - `--cpus-per-task=H`, hilos por proceso (total P·H cores);
  - `--mem`: sin límite por defecto y en general no hace falta; `--mem=0` pide toda la memoria;
  - `--gres=gpu:X`: GPUs **por nodo**, visibles para todos los procesos del nodo;
  - `--reservation=nombre`;
  - `--array=1-3`, con las variables `SLURM_ARRAY_JOB_ID`, `SLURM_ARRAY_TASK_ID`, `_COUNT`, `_MIN` y `_MAX` (https://slurm.schedmd.com/job_array.html).
- **Script:** `. /etc/profile` (entorno común y comando `module`), `module load …`, `export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK` y lanzar con `srun programa`, que reemplaza a mpirun.

### Slurm signals: `tutoriales/slurm-signals.html`
Para un cierre ordenado antes del límite:
- `#SBATCH --signal=SIGTERM@595` manda la señal 595 s antes del límite. Con el prefijo `B:` (`--signal=B:SIGTERM@…`) va al script de sbatch; sin él, al programa lanzado con `srun`.
- El programa atrapa SIGTERM (`trap … SIGTERM`), guarda su estado y sale.
- Como `srun` bloquea, después de que termina se pueden correr tareas de guardado, como copiar del /scratch al /home.

### Software preinstalado (módulos): `tutoriales/modules.html`
- **Comandos:** `module av`, `module load`, `module list` y `module purge`. Las bibliotecas aparecen después de cargar el compilador, porque la jerarquía es por compilador.
- **CUDA:** `module load nvhpc`, o `module load gcc cuda`.
- **Otros:**
  - Flags de compilación recomendados: `env | grep FLAGS`.
  - No hay Clang/LLVM.
  - Se usa OpenMPI para que todos los clusters sean homogéneos.
  - Pedidos de software, a soporte.
  - Otras versiones de compiladores, con Spack.
  - **Python: que cada usuario maneje su propio entorno en el home; se recomienda Miniconda.**

### Software con licencia: `tutoriales/licenses.html`
No hay licencias de Matlab, Ansys, Gaussian, COMSOL ni VASP. Cada usuario instala el servidor de licencias en su home y es responsable de que corra en la cabecera.

Los nodos llegan a la cabecera por un nombre interno propio de cada cluster (`head.<arquitectura>.ccad.unc.edu.ar`: rome, ivb, knl y bdw para Serafín, Mendieta, Eulogia y Mulatona); las direcciones están en la página de la wiki.

### Spack: `tutoriales/spack.html`
1. `git clone --branch releases/v0.21 https://github.com/spack/spack.git ~/spack`
2. `source ~/spack/share/spack/setup-env.sh`
3. `module load gcc spack-config`
4. `spack compiler find`
5. `spack install paquete %gcc@11.2`, dentro de tmux porque tarda.
6. `spack module tcl refresh -y`, `module use ~/spack/share/spack/modules/linux-rocky8-zen2/` y `module load …`.

En el sbatch: `source /etc/profile`, `module use …`, `module load …` y `srun --mpi=pmix programa`.

### Sesión persistente (tmux): `tutoriales/tmux.html`
`tmux` crea una sesión; `Ctrl+b d` la deja corriendo en segundo plano; `tmux a` vuelve a la última; `tmux ls` las lista; `tmux attach -t 0` se conecta a una; `Ctrl+d` la cierra. Ya está instalado en todos los clusters. Sirve para compilaciones largas en la cabecera, no para cálculos.

### Paralelizar dentro de un job: `tutoriales/parallel.html`
Usarlo sólo cuando el programa no escala para llenar un nodo.
- **GNU Parallel** (`module load parallel`): `parallel -P$NUM_JOBS 'srun -n $TASKS --cpu_bind=map_cpu:$MAP --overlap work.sh {}' ::: …`, para pasos con MPI y CPU pinning.
- **xargs:** `xargs -P ${SLURM_CPUS_PER_TASK} --max-args 1 -I{} programa {}`, para pasos independientes sin MPI.

### Monitoreo de eficiencia: `tutoriales/check-efficiency.html`
- **Slurm contabiliza lo reservado, no lo usado.**
- **Jobs terminados:** `seff JOBID` da la eficiencia de CPU y memoria; `sacct -j JOBID -o JobID,State,Elapsed,TotalCPU,MaxRSS,ReqMem…` da el detalle (`--helpformat` lista los campos). Las métricas sólo son precisas para jobs completados.
- **Jobs corriendo:** `sstat -j JOBID -o JobID,MaxRSS,AveCPU,MaxDiskRead… --allsteps`.
- **Grafana:** `jobmetrics JOBID [--follow]` da un enlace al dashboard de https://stats.ccad.unc.edu.ar.

### Directorios compartidos: `tutoriales/shared-dir.html`
Un grupo y `/home/shared/<grupo>` se piden por mail a soporte, indicando clusters, nombre, usuarios y notas.

### Gestión de archivos: `tutoriales/archivos/*.html`
Todos los métodos requieren las claves SSH configuradas.
- **MobaXterm (SFTP, Windows):** activar el agente interno "MobaAgent", crear una sesión SFTP con usuario y `cluster.ccad.unc.edu.ar`.
- **VS Code:** Remote-SSH con "Agregar nuevo host SSH", por ejemplo `ssh $USUARIO@nabucodonosor.ccad.unc.edu.ar` (el ejemplo es de un cluster ya retirado).
- **GNOME/KDE:** "Otras ubicaciones → Conectarse al servidor" con `ssh://usuario@cluster.ccad.unc.edu.ar`.
- **SSHFS:** `sshfs $USUARIO@mendieta.ccad.unc.edu.ar:/home/$USUARIO ~/datos_mendieta`; se desmonta con `fusermount -u`.
- **FSTAB:** montaje automático con `fuse.sshfs`, `user_allow_other` en `/etc/fuse.conf` e `identityfile=…`.
- **SCP:** `scp archivo $USUARIO@cluster:~/destino/`; `-r` para directorios; también entre dos clusters.
- **Rsync:** `rsync -avz origen $USUARIO@cluster:~/destino/` para subir, al revés para bajar; también entre clusters.

## Programas

### NextFlow: `programas/nextflow.html`
- Se instala con Spack (`spack install nextflow`).
- En el sbatch, el directorio de trabajo va en el /scratch: `NXF_WORK`, `NXF_TEMP` y `NXF_LAUNCHER` en `/scratch/$USER/…`.
- **Contenedores Apptainer:** **sólo la cabecera tiene internet**, así que las imágenes se bajan antes con `apptainer pull docker://…` a un directorio del home, con `APPTAINER_CACHEDIR` y `NXF_APPTAINER_CACHEDIR` en el home. Después: `nextflow run … -with-apptainer imagen.sif`.

### Quantum Espresso: `programas/qe.html`
- **Datos:** los esenciales en el /home; los temporales en el /scratch local (`ESPRESSO_TMPDIR=/scratch/$USER/$SLURM_JOB_ID`), copiando la entrada al empezar y la salida al terminar con `srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 rsync …`.
- **Paralelismo:** `-npool`, `-nimage`, `-ndiag`, `-ntg`, `-nb` y `-nt`, más benchmarking (speedup, eficiencia; menos de 50-60 % indica sobreescalado).
- **Patrón de reinicio:** `max_seconds` por debajo del límite de la cola y `restart_mode='restart'`. "Sólo basta volver a enviar el script a la cola sin realizar ninguna modificación", incluso en otra cantidad de nodos.

## Ayuda

### Contactar a soporte: `ayuda/soporte.html`
Primero buscar en la wiki y en el Zulip (https://ccadunc.zulipchat.com/). Después, mail a `soporte@ccad.unc.edu.ar` con usuario y cluster, directorio, script de Slurm y archivo con el error completo. Para una duda puntual, reunión por Google Meet: https://calendly.com/ccad-soporte-ale/consulta

### Enviar sugerencias: `ayuda/colaborar.html`
Issues o merge requests en https://gitlab.com/ccad/wiki-rtd

## Sitio institucional (supercomputo.unc.edu.ar)
- `/servicios/`: servicios para investigadores del Sistema Científico Nacional, pedidos de proyecto a director{arroba}ccad.unc.edu.ar, evaluados por un Comité Científico.
- `/servicios/tarifario/`: costo de la hora-core; la resolución RR-469-2021 habilita el cobro.
- `/servicios/pedido-de-uso-intensivo-ventanilla-permanente/`: períodos de cómputo con exclusividad, por formulario.
- `/equipamiento/`: en producción Boogie (2025), Gordito (2025/2026), Mendieta Fase 2 (2022), Serafín (2021), Eulogia (2018/2021) y Mulatona (2018); Nabucodonosor, retirado. Se está montando Helvetios (2026).
- No se encontraron normas de uso publicadas más allá de la wiki. El estatuto y lo aceptado en el formulario de alta no se revisaron.
