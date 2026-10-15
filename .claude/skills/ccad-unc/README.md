# Skill `ccad-unc`

## Qué es

Una skill de Claude Code para trabajar con los clusters de **UNC Supercómputo (CCAD, Universidad Nacional de Córdoba)**: Mendieta (GPU A30), Serafín, Eulogia y Mulatona. Le da a Claude el contexto para:

- conectarse por SSH y usar Slurm (`sbatch`, `squeue`, `sacct`, `seff`);
- elegir partición y tiempo (`short` hasta 1 h, `multi` hasta 2 días);
- preparar un entorno de Python/PyTorch que funcione en los nodos;
- mover archivos (`rsync`) y traer resultados;
- diagnosticar errores típicos (`TIMEOUT`, `DependencyNeverSatisfied`, imports lentos).

Contenido:

| Archivo | Qué tiene |
|---|---|
| `SKILL.md` | Reglas del CCAD y flujo de trabajo. Es lo que Claude carga al usar la skill. |
| `references/wiki.md` | Resumen, página por página, de la wiki oficial (https://wiki.ccad.unc.edu.ar/). |
| `references/lecciones-mendieta.md` | Lo aprendido en la práctica y que no está en la wiki (entorno en `/scratch`, `set -u`, `singleton`, retoma en GPU, parar a tiempo, SIGTERM, etc.). |
| `references/plantilla-sbatch.sh` | Plantilla mínima de un job de GPU. |

**Límite actual:** la plantilla y las lecciones nombran tres scripts que viven en el repo land2vec (`scripts/cluster/entorno_scratch.sh`, `empaquetar_entorno.sh`, `prueba_mendieta.sbatch`). Fuera de ese repo hay que copiarlos junto a la skill o reescribir esas rutas.

## Cómo instalarla

La carpeta de la skill es `ccad-unc/` (la que contiene este README y `SKILL.md`). Claude Code solo busca skills directamente dentro de `.claude/skills/`, en el proyecto o en tu home. Ejemplos con el repo `skills_agents`, donde la skill está en `skills/ccad-unc`.

### Caso 1: proyecto con contenedor que ya existe (sin rebuild)

No se toca el contenedor ni su configuración. En la terminal, en la raíz del proyecto:

```bash
mkdir -p .claude/skills
git clone git@github.com:<usuario>/skills_agents.git .claude/skills_agents
ln -s ../skills_agents/skills/ccad-unc .claude/skills/ccad-unc
printf ".claude/skills_agents/\n.claude/skills/ccad-unc\n" >> .gitignore
```

Para actualizar cuando la skill cambie: `git -C .claude/skills_agents pull`.

Alternativa sin enlace: copiar la carpeta (`cp -r .../skills/ccad-unc .claude/skills/`). Queda versionada con el proyecto, pero hay que recopiarla a mano en cada cambio.

### Caso 2: proyecto nuevo (se arma el devcontainer desde cero)

En `.devcontainer/devcontainer.json`:

```json
{
  "image": "mcr.microsoft.com/devcontainers/base:ubuntu",
  "features": {
    "ghcr.io/devcontainers/features/node:2.1.0": {},
    "ghcr.io/anthropics/devcontainer-features/claude-code:1.0": {}
  },
  "mounts": ["source=claude-config,target=/home/vscode/.claude,type=volume"],
  "postCreateCommand": "mkdir -p ~/.claude/skills && (git clone git@github.com:<usuario>/skills_agents.git ~/skills_agents || git -C ~/skills_agents pull) && ln -sfn ~/skills_agents/skills/ccad-unc ~/.claude/skills/ccad-unc"
}
```

Después, `Dev Containers: Reopen in Container`. La skill queda en `~/.claude/skills` y vale para todos los proyectos de ese contenedor. El volumen `claude-config` conserva la memoria de Claude entre rebuilds. Para actualizar: `git -C ~/skills_agents pull`.

### Caso 3: sin contenedor (máquina propia)

```bash
git clone git@github.com:<usuario>/skills_agents.git ~/skills_agents
mkdir -p ~/.claude/skills
ln -s ~/skills_agents/skills/ccad-unc ~/.claude/skills/ccad-unc
```

### Verificar la instalación

```bash
ls -la .claude/skills/        # o ~/.claude/skills/ según el caso
```

Debe aparecer `ccad-unc -> ...` apuntando a una carpeta que exista. Después, en una sesión nueva de Claude Code, `ccad-unc` figura entre las skills disponibles.

Notas:
- Si el repo `skills_agents` es privado, el clon por SSH necesita tu agente SSH. VS Code lo reenvía al contenedor por defecto.
- Un rebuild borra el home del contenedor (`~/.claude`, incluida la memoria de Claude) si no hay volumen montado. Con el Caso 1 la skill sobrevive igual, porque está dentro del proyecto.

## Cómo usarla en una sesión de chat

1. **Invocarla.** Escribí `/ccad-unc`, o pedí algo del cluster con palabras normales ("cómo encolo un job en Mendieta", "revisá cómo viene mi job"); Claude carga la skill solo cuando el tema coincide.
2. **Conexión.** Claude se conecta con tu agente SSH (`ssh -o BatchMode=yes <usuario>@mendieta.ccad.unc.edu.ar`). La clave privada nunca se le pide ni se copia. Si el agente no está disponible, no puede conectarse.
3. **Qué hace sin preguntar:** comandos de lectura (`squeue`, `sacct`, `sinfo`, leer logs y archivos de resultados, `sbatch --test-only`).
4. **Qué te pregunta antes:** cualquier cosa que modifique el cluster: encolar (`sbatch`), cancelar (`scancel`), cambiar jobs (`scontrol update`), subir o sobrescribir archivos.
5. **Pedidos típicos:**
   - "Preparame un sbatch para entrenar X en una GPU" (usa `references/plantilla-sbatch.sh`).
   - "Estimá cuándo arranca este job en multi" (`squeue --me --start`).
   - "El job terminó en TIMEOUT, ¿por qué?" (revisa log y `sacct`).
   - "Bajá los resultados a mi máquina" (`rsync`).
6. **Seguimiento periódico:** se puede combinar con `/loop` para revisar un job cada cierto tiempo, y avisar solo si algo falla o termina.

Antes del primer job conviene pedirle a Claude que lea `references/lecciones-mendieta.md`: cada punto costó al menos un job fallido.
