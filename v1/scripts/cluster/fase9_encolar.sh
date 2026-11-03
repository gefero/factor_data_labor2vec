#!/usr/bin/env bash
# Fase 9, etapa B: encola en Mendieta los jobs de extracción de vectores. Se corre en la máquina local, desde la raíz del repo.
#
#   MAIL=correo bash scripts/cluster/fase9_encolar.sh b1|extras|ctrl|b2|b3 [--probar]
#
#   b1      estabilidad entre semillas: vectores de pre_150ep_s1/_s2 y de los ajustes de T1 y T3 con semillas 1 y 2 (6 jobs)
#   extras  pre_150ep_sinuv (sin rama, provincia ni antigüedad) y un encoder con pesos sin entrenar (2 jobs)
#   ctrl    pre con las fechas permutadas y con el orden permutado (2 jobs; val y test, para repetir las sondas)
#   b2      pre con la historia cortada en 2010-12, 2012-12 y 2014-12 (3 jobs, sólo test)
#   b3      intervenciones sobre la historia con los modelos t1, t3 y pre (3 jobs; cada uno hace 5 pasadas)
#
# Antes de encolar sube el código (NO_SUBIR=1 lo omite) y comprueba que existan todos los modelos en el cluster. `--probar`
# sólo imprime los comandos. Los resultados quedan en ~/labor2vec/v1/runs/fase9/. Todo job lleva aviso por mail (BEGIN, END, FAIL).
set -euo pipefail
cd "$(dirname "$0")/../.."

GRUPO=${1:?b1 | extras | ctrl | b2 | b3}
PROBAR=${2:-}
USUARIO=${USUARIO:-grosati}
HOST=${HOST:-mendieta.ccad.unc.edu.ar}
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 $USUARIO@$HOST"
[ -n "$PROBAR" ] || : "${MAIL:?falta MAIL (correo para los avisos de Slurm)}"

T1=configs/ajuste_t1.yaml; T3=configs/ajuste_t3.yaml
PRE=runs/pre_150ep/modelo_final.pt
# formato: script|nombre|config|modelo|argumentos de extraer
case "$GRUPO" in
b1) JOBS=(
    "personas|pre_s1|$T1|runs/pre_150ep_s1/modelo_final.pt|--sin-cabeza --sin-grad"
    "personas|pre_s2|$T1|runs/pre_150ep_s2/modelo_final.pt|--sin-cabeza --sin-grad"
    "personas|t1_s1|$T1|runs/t1_l2v_lr0001_s1/modelo_final.pt|--sin-grad"
    "personas|t1_s2|$T1|runs/t1_l2v_lr0001_s2/modelo_final.pt|--sin-grad"
    "personas|t3_s1|$T3|runs/t3_l2v_s1/modelo_final.pt|--sin-grad"
    "personas|t3_s2|$T3|runs/t3_l2v_s2/modelo_final.pt|--sin-grad") ;;
extras) JOBS=(
    "personas|pre_sinuv|$T1|runs/pre_150ep_sinuv/modelo_final.pt|--sin-cabeza --sin-grad --set datos.atributos=sin_ultimo_valor"
    "personas|pre_azar|$T1|$PRE|--sin-cabeza --sin-grad --azar") ;;
ctrl) JOBS=(
    "personas|pre_ctrl_tiempo|$T1|$PRE|--sin-cabeza --sin-grad --set datos.control_temporal=tiempo"
    "personas|pre_ctrl_orden|$T1|$PRE|--sin-cabeza --sin-grad --set datos.control_temporal=orden") ;;
b2) JOBS=(
    "personas|pre_f179|$T1|$PRE|--sin-cabeza --sin-grad --solo-test --set datos.fin_ventana=179"
    "personas|pre_f203|$T1|$PRE|--sin-cabeza --sin-grad --solo-test --set datos.fin_ventana=203"
    "personas|pre_f227|$T1|$PRE|--sin-cabeza --sin-grad --solo-test --set datos.fin_ventana=227") ;;
b3) JOBS=(
    "intervenciones|t1|$T1|runs/t1_l2v_lr0001/modelo_final.pt|"
    "intervenciones|t3|$T3|runs/t3_l2v_s0/modelo_final.pt|"
    "intervenciones|pre|$T1|$PRE|--sin-cabeza") ;;
*) echo "grupo desconocido: $GRUPO" >&2; exit 1 ;;
esac

if [ -z "$PROBAR" ]; then
    echo "== Comprobando los modelos en el cluster"
    for j in "${JOBS[@]}"; do
        IFS='|' read -r _ nombre _ modelo _ <<<"$j"
        $SSH "test -s ~/labor2vec/v1/$modelo" || { echo "NO EXISTE en el cluster: $modelo (job $nombre)" >&2; exit 1; }
    done
    [ -n "${NO_SUBIR:-}" ] || { echo "== Subiendo el código"; bash scripts/cluster/sincronizar_mendieta.sh subir "$USUARIO"; }
fi

echo "== Encolando ($GRUPO)"
for j in "${JOBS[@]}"; do
    IFS='|' read -r script nombre config modelo args <<<"$j"
    cmd="cd ~/labor2vec/v1 && BASE=runs/fase9 sbatch --parsable --mail-type=BEGIN,END,FAIL --mail-user=${MAIL:-CORREO} -J f9_${script:0:3}_$nombre scripts/cluster/$script.sbatch $nombre $config $modelo $args"
    if [ -n "$PROBAR" ]; then echo "$cmd"; else echo "$nombre -> job $($SSH "$cmd")"; fi
done
