#!/bin/bash
set -euo pipefail

DATASET_FILE="${DATASET:-dados_50k.csv}"
SAMPLES="${SAMPLES:-50000}"
SEED="${SEED:-42}"
N_LIST="${N_LIST:-5000 10000 20000 40000}"
THREADS_LIST="${THREADS_LIST:-1 2 4 8 10 16 20 32 40}"

echo "=== KNN benchmark NUMA ==="
echo "Dataset: $DATASET_FILE"
echo "Amostras geradas: $SAMPLES"
echo "Tamanhos medidos: $N_LIST"
echo "Threads: $THREADS_LIST"

make all generate-synthetic

if [ ! -f "$DATASET_FILE" ]; then
    echo "Gerando dataset uma única vez..."
    ./build/generate_synthetic "$DATASET_FILE" "$SAMPLES" "$SEED"
else
    echo "Dataset existente; geração será pulada."
fi

export OMP_PLACES="${OMP_PLACES:-cores}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-spread}"
export OMP_DISPLAY_AFFINITY="${OMP_DISPLAY_AFFINITY:-TRUE}"
export LC_NUMERIC="${LC_NUMERIC:-C}"

echo "OMP_PLACES=$OMP_PLACES"
echo "OMP_PROC_BIND=$OMP_PROC_BIND"

DATASET="$DATASET_FILE" ./benchmark.sh "$N_LIST" "$THREADS_LIST"
