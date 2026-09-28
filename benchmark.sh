#!/bin/bash
# ===========================================================================
# Script de benchmark para o KNN OpenMP
# Testa: sequencial vs variantes paralelas (schedule x simd) x threads x N
# Uso: ./benchmark.sh [N_list] [threads_list]
# ===========================================================================

DATASET="${DATASET:-dados_50k.csv}"
K="${K:-5}"
N_LIST="${1:-5000 10000 20000 40000}"
THREADS_LIST="${2:-1 2 4 8 10 16 20 32 40}"
RESULTS_FILE="${RESULTS_FILE:-resultados_knn.csv}"
B="build"

# Binarios paralelos: binario:schedule:simd_label
VARIANTS="knn_par_static_simd:static:ON
knn_par_static_nosimd:static:OFF
knn_par_dynamic_simd:dynamic:ON
knn_par_dynamic_nosimd:dynamic:OFF
knn_par_guided_simd:guided:ON
knn_par_guided_nosimd:guided:OFF"

# Compilar
make all 2>/dev/null || { echo "Erro na compilacao"; exit 1; }

if [ ! -f "$DATASET" ]; then
    echo "Erro: dataset nao encontrado: $DATASET" >&2
    echo "Gere-o com: ./build/generate_synthetic dados_50k.csv 50000 42" >&2
    exit 1
fi

# Header CSV
echo "N,num_train,num_test,versao,schedule,simd,threads,tempo_seg,acuracia" > $RESULTS_FILE

for N in $N_LIST; do
    NUM_TRAIN=$(echo "$N * 80 / 100" | bc)
    NUM_TEST=$((N - NUM_TRAIN))
    echo "=========================================="
    echo "N=$N amostras (treino=$NUM_TRAIN, teste=$NUM_TEST)"
    echo "=========================================="

    # 1. Sequencial (baseline)
    OUTPUT=$($B/knn_seq $DATASET $K $N 2>&1) || {
        echo "Erro ao executar versao sequencial (dataset=$DATASET, N=$N):" >&2
        echo "$OUTPUT" >&2
        exit 1
    }
    SEQ_TIME=$(echo "$OUTPUT" | grep "Tempo Sequencial" | awk '{print $3}')
    ACURACIA=$(echo "$OUTPUT" | grep "Acuracia" | awk '{print $2}' | tr -d '%')
    echo "  SEQ:                     ${SEQ_TIME}s  acc=${ACURACIA}%"
    echo "$N,$NUM_TRAIN,$NUM_TEST,sequencial,-,-,1,$SEQ_TIME,$ACURACIA" >> $RESULTS_FILE

    # 2. Variantes paralelas
    for ENTRY in $VARIANTS; do
        BIN=$(echo "$ENTRY" | cut -d: -f1)
        SCHED=$(echo "$ENTRY" | cut -d: -f2)
        SIMD_LABEL=$(echo "$ENTRY" | cut -d: -f3)

        for NT in $THREADS_LIST; do
            OUTPUT=$(OMP_NUM_THREADS=$NT $B/$BIN $DATASET $K $N 2>&1) || {
                echo "Erro ao executar $BIN com $NT threads (dataset=$DATASET, N=$N):" >&2
                echo "$OUTPUT" >&2
                exit 1
            }
            TEMPO=$(echo "$OUTPUT" | grep "Tempo Paralelo" | awk -F': ' '{print $2}' | awk '{print $1}')
            ACURACIA=$(echo "$OUTPUT" | grep "Acuracia" | awk '{print $2}' | tr -d '%')

            if [ -n "$SEQ_TIME" ] && [ -n "$TEMPO" ]; then
                SPEEDUP=$(echo "scale=2; $SEQ_TIME / $TEMPO" | bc 2>/dev/null || echo "N/A")
            else
                SPEEDUP="N/A"
            fi
            printf "  %-8s simd=%-3s %2dT: %10ss  acc=%s%%  speedup=%sx\n" \
                   "$SCHED" "$SIMD_LABEL" "$NT" "$TEMPO" "$ACURACIA" "$SPEEDUP"
            echo "$N,$NUM_TRAIN,$NUM_TEST,paralelo,$SCHED,$SIMD_LABEL,$NT,$TEMPO,$ACURACIA" >> $RESULTS_FILE
        done
    done
    echo ""
done

echo "=========================================="
echo "Resultados salvos em: $RESULTS_FILE"
echo "=========================================="
