#!/bin/bash
# Carregar VTune ANTES do set -u (vtune-vars.sh usa variaveis nao definidas)
source /home/intel/oneapi/vtune/2021.1.1/vtune-vars.sh 2>/dev/null || true

set -euo pipefail

DATASET_FILE="${DATASET:-dados_50k.csv}"
SAMPLES="${SAMPLES:-50000}"
SEED="${SEED:-42}"
DEFAULT_N_LIST="5000 10000 20000 40000"
DEFAULT_THREADS_LIST="1 2 4 8 10 16 20 32 40"

usage() {
    cat <<'EOF'
Uso:
  ./run_all_50k.sh
  ./run_all_50k.sh "5000 40000" "20 40"
  ./run_all_50k.sh --vtune
  ./run_all_50k.sh --vtune "40000" "20 40"

Sem argumentos: todos os tamanhos e threads.
Argumento 1: lista de N.
Argumento 2: lista de threads.
--vtune: executa também coletas VTune para os casos selecionados.
EOF
}

VTUNE=0
if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    usage
    exit 0
fi
if [ "${1:-}" = "--vtune" ]; then
    VTUNE=1
    shift
fi

N_LIST="${1:-${N_LIST:-$DEFAULT_N_LIST}}"
THREADS_LIST="${2:-${THREADS_LIST:-$DEFAULT_THREADS_LIST}}"

make all generate-synthetic
if [ ! -f "$DATASET_FILE" ]; then
    ./build/generate_synthetic "$DATASET_FILE" "$SAMPLES" "$SEED"
fi

export OMP_PLACES="${OMP_PLACES:-cores}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-spread}"
export OMP_DISPLAY_AFFINITY="${OMP_DISPLAY_AFFINITY:-TRUE}"
export LC_NUMERIC="${LC_NUMERIC:-C}"

echo "Dataset: $DATASET_FILE"
echo "N: $N_LIST"
echo "Threads: $THREADS_LIST"

DATASET="$DATASET_FILE" ./benchmark.sh "$N_LIST" "$THREADS_LIST"

if [ "$VTUNE" -eq 1 ]; then
    if ! command -v vtune >/dev/null 2>&1; then
        echo "Erro: --vtune foi solicitado, mas vtune nao esta no PATH" >&2
        exit 1
    fi

    VTUNE_DIR="${VTUNE_DIR:-vtune-dados-50k}"
    VTUNE_REPORTS="$VTUNE_DIR/reports"
    mkdir -p "$VTUNE_DIR" "$VTUNE_REPORTS"
    VARIANTS="knn_par_static_simd:static:ON
knn_par_static_nosimd:static:OFF
knn_par_dynamic_simd:dynamic:ON
knn_par_dynamic_nosimd:dynamic:OFF
knn_par_guided_simd:guided:ON
knn_par_guided_nosimd:guided:OFF"

    for N in $N_LIST; do
        tag="seq_N${N}"
        vtune -collect hotspots -knob sampling-mode=hw -result-dir "$VTUNE_DIR/hotspots-$tag" \
          -- ./build/knn_seq "$DATASET_FILE" 5 "$N"
        vtune -report hotspots -r "$VTUNE_DIR/hotspots-$tag" -format=csv > "$VTUNE_REPORTS/hotspots-${tag}.csv" 2>&1
        vtune -report summary -r "$VTUNE_DIR/hotspots-$tag" -format=csv > "$VTUNE_REPORTS/summary-${tag}.csv" 2>&1

        while IFS=: read -r bin sched simd; do
            [ -n "$bin" ] || continue
            for NT in $THREADS_LIST; do
                tag="par_${sched}_simd${simd}_${NT}T_N${N}"
                vtune -collect hotspots -knob sampling-mode=hw -result-dir "$VTUNE_DIR/hotspots-$tag" \
                  -- env OMP_NUM_THREADS="$NT" "$PWD/build/$bin" "$DATASET_FILE" 5 "$N"
                vtune -report hotspots -r "$VTUNE_DIR/hotspots-$tag" -format=csv > "$VTUNE_REPORTS/hotspots-${tag}.csv" 2>&1
                vtune -report summary -r "$VTUNE_DIR/hotspots-$tag" -format=csv > "$VTUNE_REPORTS/summary-${tag}.csv" 2>&1
            done
        done <<EOF
$VARIANTS
EOF
    done

    echo ""
    echo "Reports VTune em CSV salvos em: $VTUNE_REPORTS/"
    echo "Total: $(ls "$VTUNE_REPORTS"/*.csv 2>/dev/null | wc -l) arquivos"
fi
