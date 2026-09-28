#!/bin/bash
#SBATCH --job-name=knn-openmp
#SBATCH --partition=hype
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# ===========================================================================
# KNN OpenMP - Script de benchmark para o PCAD (particao hype)
# Maquinas hype: 2 x Intel Xeon E5-2650 v3, 20 cores / 40 threads, 128 GB
#
# Testa:
#   - Versao sequencial (baseline)
#   - 6 variantes paralelas: schedule(static|dynamic|guided) x simd(on|off)
#   - Threads: 1, 2, 4, 8, 16, 20, 40
#   - Tamanho da entrada (N): 100, 200, 300, 400, 569
# ===========================================================================

echo "============================================"
echo "Job: $SLURM_JOB_NAME (ID: $SLURM_JOB_ID)"
echo "No: $(hostname)"
echo "Data: $(date)"
echo "CPUs alocadas: $SLURM_CPUS_PER_TASK"
echo "============================================"
echo ""

# Copiar dados para o scratch (disco local)
mkdir -p $SCRATCH/knn-openmp
cp $HOME/knn-openmp/KNNAlgorithmDataset.csv $SCRATCH/knn-openmp/
cp $HOME/knn-openmp/knn_sequencial.c $SCRATCH/knn-openmp/
cp $HOME/knn-openmp/knn_parallel.c $SCRATCH/knn-openmp/
cp $HOME/knn-openmp/Makefile $SCRATCH/knn-openmp/
cd $SCRATCH/knn-openmp

# Compilar
echo "Compilando..."
make clean && make all
echo ""

# ---------------------------------------------------------------------------
# Parametros do experimento
# ---------------------------------------------------------------------------
DATASET="KNNAlgorithmDataset.csv"
K=5
N_LIST="100 200 300 400 569"
THREADS_LIST="1 2 4 8 16 20 40"
RESULTS_FILE="resultados_knn.csv"

BINS="knn_par_static_simd knn_par_static_nosimd knn_par_dynamic_simd knn_par_dynamic_nosimd knn_par_guided_simd knn_par_guided_nosimd"

echo "N,num_train,num_test,versao,schedule,simd,threads,tempo_seg,acuracia" > $RESULTS_FILE

# ---------------------------------------------------------------------------
# Executar experimentos
# ---------------------------------------------------------------------------
for N in $N_LIST; do
    NUM_TRAIN=$((N * 80 / 100))
    NUM_TEST=$((N - NUM_TRAIN))
    echo "=========================================="
    echo "N=$N amostras (treino=$NUM_TRAIN, teste=$NUM_TEST)"
    echo "=========================================="

    # 1. Sequencial (baseline)
    echo "  [SEQ] Executando..."
    OUTPUT=$(build/knn_seq $DATASET $K $N)
    echo "$OUTPUT"
    TEMPO=$(echo "$OUTPUT" | grep "Tempo Sequencial" | awk '{print $3}')
    ACURACIA=$(echo "$OUTPUT" | grep "Acuracia" | awk '{print $2}' | tr -d '%')
    echo "$N,$NUM_TRAIN,$NUM_TEST,sequencial,-,-,1,$TEMPO,$ACURACIA" >> $RESULTS_FILE
    echo ""

    # 2. Variantes paralelas
    for BIN in $BINS; do
        SCHED=$(echo $BIN | sed 's/knn_par_//' | cut -d_ -f1)
        SIMD_RAW=$(echo $BIN | sed 's/knn_par_//' | cut -d_ -f2)
        if [ "$SIMD_RAW" = "simd" ]; then SIMD_LABEL="ON"; else SIMD_LABEL="OFF"; fi

        for NT in $THREADS_LIST; do
            echo "  [$SCHED simd=$SIMD_LABEL ${NT}T] Executando..."
            OUTPUT=$(OMP_NUM_THREADS=$NT build/$BIN $DATASET $K $N)
            TEMPO=$(echo "$OUTPUT" | grep "Tempo Paralelo" | awk -F': ' '{print $2}' | awk '{print $1}')
            ACURACIA=$(echo "$OUTPUT" | grep "Acuracia" | awk '{print $2}' | tr -d '%')
            echo "    Tempo: ${TEMPO}s  Acuracia: ${ACURACIA}%"
            echo "$N,$NUM_TRAIN,$NUM_TEST,paralelo,$SCHED,$SIMD_LABEL,$NT,$TEMPO,$ACURACIA" >> $RESULTS_FILE
        done
        echo ""
    done
done

echo ""
echo "=========================================="
echo "Resultados salvos em: $SCRATCH/knn-openmp/$RESULTS_FILE"
echo "=========================================="

# Copiar resultados de volta
cp $RESULTS_FILE $HOME/knn-openmp/
echo "Resultados copiados para: $HOME/knn-openmp/$RESULTS_FILE"
echo "Job finalizado em: $(date)"
