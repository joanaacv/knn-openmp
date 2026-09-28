#!/bin/bash
# ===========================================================================
# Script completo: limpa resultados anteriores, roda benchmark + VTune
# para TODAS as combinacoes e organiza tudo em resultados/ para download.
#
# Uso: ./run_all.sh
# ===========================================================================

set -e

DATASET="KNNAlgorithmDataset.csv"
K=5
N_LIST="100 200 300 400 569"
THREADS_LIST="1 2 4 8 16 20 40"
RESULTS_DIR="resultados"
VTUNE_DIR="vtune-dados"
B="build"

VARIANTS="knn_par_static_simd:static:ON
knn_par_static_nosimd:static:OFF
knn_par_dynamic_simd:dynamic:ON
knn_par_dynamic_nosimd:dynamic:OFF
knn_par_guided_simd:guided:ON
knn_par_guided_nosimd:guided:OFF"

echo "============================================"
echo "KNN OpenMP - Execucao completa"
echo "Maquina: $(hostname)"
echo "Data: $(date)"
echo "============================================"
echo ""

# ---------------------------------------------------------------------------
# 1. Limpar tudo
# ---------------------------------------------------------------------------
echo "[1/5] Limpando resultados anteriores..."
rm -rf $VTUNE_DIR
rm -rf $RESULTS_DIR
rm -f resultados_knn.csv
echo "  Limpo."
echo ""

# ---------------------------------------------------------------------------
# 2. Recompilar
# ---------------------------------------------------------------------------
echo "[2/5] Compilando..."
make clean && make all
echo "  Compilado."
echo ""

# ---------------------------------------------------------------------------
# 3. Benchmark completo (sem VTune)
# ---------------------------------------------------------------------------
echo "[3/5] Rodando benchmark completo..."
echo ""

RESULTS_FILE="resultados_knn.csv"
echo "N,num_train,num_test,versao,schedule,simd,threads,tempo_seg,acuracia" > $RESULTS_FILE

for N in $N_LIST; do
    NUM_TRAIN=$((N * 80 / 100))
    NUM_TEST=$((N - NUM_TRAIN))
    echo "  N=$N (treino=$NUM_TRAIN, teste=$NUM_TEST)"

    # Sequencial
    OUTPUT=$($B/knn_seq $DATASET $K $N 2>&1)
    SEQ_TIME=$(echo "$OUTPUT" | grep "Tempo Sequencial" | awk '{print $3}')
    ACURACIA=$(echo "$OUTPUT" | grep "Acuracia" | awk '{print $2}' | tr -d '%')
    echo "    SEQ: ${SEQ_TIME}s  acc=${ACURACIA}%"
    echo "$N,$NUM_TRAIN,$NUM_TEST,sequencial,-,-,1,$SEQ_TIME,$ACURACIA" >> $RESULTS_FILE

    # Paralelo
    for ENTRY in $VARIANTS; do
        BIN=$(echo "$ENTRY" | cut -d: -f1)
        SCHED=$(echo "$ENTRY" | cut -d: -f2)
        SIMD_LABEL=$(echo "$ENTRY" | cut -d: -f3)

        for NT in $THREADS_LIST; do
            OUTPUT=$(OMP_NUM_THREADS=$NT $B/$BIN $DATASET $K $N 2>&1)
            TEMPO=$(echo "$OUTPUT" | grep "Tempo Paralelo" | awk -F': ' '{print $2}' | awk '{print $1}')
            ACURACIA=$(echo "$OUTPUT" | grep "Acuracia" | awk '{print $2}' | tr -d '%')
            echo "$N,$NUM_TRAIN,$NUM_TEST,paralelo,$SCHED,$SIMD_LABEL,$NT,$TEMPO,$ACURACIA" >> $RESULTS_FILE
        done
    done
    echo "    Paralelo concluido para N=$N"
done
echo ""
echo "  Benchmark salvo em: $RESULTS_FILE"
echo ""

# ---------------------------------------------------------------------------
# 4. VTune - TODAS as combinacoes
# ---------------------------------------------------------------------------
echo "[4/5] Rodando analises VTune para TODAS as combinacoes..."
echo ""

source /home/intel/oneapi/vtune/2021.1.1/vtune-vars.sh 2>/dev/null

mkdir -p $VTUNE_DIR

# Para cada N
for N in $N_LIST; do
    echo "  === VTune N=$N ==="

    # -----------------------------------------------------------------------
    # 4a. Sequencial
    # -----------------------------------------------------------------------
    TAG="seq_N${N}"
    echo "    [SEQ] performance-snapshot..."
    vtune -collect performance-snapshot -result-dir $VTUNE_DIR/perf-$TAG \
          -- $B/knn_seq $DATASET $K $N > /dev/null 2>&1
    vtune -report summary -r $VTUNE_DIR/perf-$TAG > $VTUNE_DIR/perf-${TAG}.txt 2>&1

    echo "    [SEQ] hotspots..."
    vtune -collect hotspots -knob sampling-mode=hw -result-dir $VTUNE_DIR/hotspots-$TAG \
          -- $B/knn_seq $DATASET $K $N > /dev/null 2>&1
    vtune -report hotspots -r $VTUNE_DIR/hotspots-$TAG > $VTUNE_DIR/hotspots-${TAG}.txt 2>&1

    # -----------------------------------------------------------------------
    # 4b. Paralelo - todas as variantes x todas as threads
    # -----------------------------------------------------------------------
    for ENTRY in $VARIANTS; do
        BIN=$(echo "$ENTRY" | cut -d: -f1)
        SCHED=$(echo "$ENTRY" | cut -d: -f2)
        SIMD_LABEL=$(echo "$ENTRY" | cut -d: -f3)

        for NT in $THREADS_LIST; do
            TAG="par_${SCHED}_simd${SIMD_LABEL}_${NT}T_N${N}"
            echo "    [PAR] $SCHED simd=$SIMD_LABEL ${NT}T - performance-snapshot..."
            vtune -collect performance-snapshot -result-dir $VTUNE_DIR/perf-$TAG \
                  -- env OMP_NUM_THREADS=$NT $B/$BIN $DATASET $K $N > /dev/null 2>&1
            vtune -report summary -r $VTUNE_DIR/perf-$TAG > $VTUNE_DIR/perf-${TAG}.txt 2>&1

            echo "    [PAR] $SCHED simd=$SIMD_LABEL ${NT}T - hotspots..."
            vtune -collect hotspots -knob sampling-mode=hw -result-dir $VTUNE_DIR/hotspots-$TAG \
                  -- env OMP_NUM_THREADS=$NT $B/$BIN $DATASET $K $N > /dev/null 2>&1
            vtune -report hotspots -r $VTUNE_DIR/hotspots-$TAG > $VTUNE_DIR/hotspots-${TAG}.txt 2>&1

            echo "    [PAR] $SCHED simd=$SIMD_LABEL ${NT}T - hpc-performance..."
            vtune -collect hpc-performance -result-dir $VTUNE_DIR/hpc-$TAG \
                  -- env OMP_NUM_THREADS=$NT $B/$BIN $DATASET $K $N > /dev/null 2>&1
            vtune -report summary -r $VTUNE_DIR/hpc-$TAG > $VTUNE_DIR/hpc-${TAG}.txt 2>&1
        done
    done
    echo ""
done

echo "  VTune concluido."
echo ""

# ---------------------------------------------------------------------------
# 5. Organizar tudo na pasta resultados/
# ---------------------------------------------------------------------------
echo "[5/5] Organizando resultados..."

mkdir -p $RESULTS_DIR/vtune

# CSV do benchmark
cp $RESULTS_FILE $RESULTS_DIR/

# Relatorios VTune em texto
cp $VTUNE_DIR/*.txt $RESULTS_DIR/vtune/

# Codigo fonte
cp knn_sequencial.c $RESULTS_DIR/
cp knn_parallel.c $RESULTS_DIR/
cp Makefile $RESULTS_DIR/

# Contar resultados
N_VTUNE_FILES=$(ls $RESULTS_DIR/vtune/*.txt 2>/dev/null | wc -l)

echo ""
echo "============================================"
echo "CONCLUIDO!"
echo "============================================"
echo ""
echo "Pasta com todos os resultados: ~/knn-openmp/$RESULTS_DIR/"
echo "  - resultados_knn.csv: benchmark (tempos, speedup)"
echo "  - vtune/: $N_VTUNE_FILES relatorios VTune em texto"
echo "  - codigo fonte: knn_sequencial.c, knn_parallel.c, Makefile"
echo ""
echo "Conteudo:"
ls -la $RESULTS_DIR/
echo ""
echo "VTune:"
ls $RESULTS_DIR/vtune/ | head -20
echo "  ... (total: $N_VTUNE_FILES arquivos)"
echo ""
echo "Para baixar para o Mac:"
echo "  rsync -avP jacvargas@gppd-hpc.inf.ufrgs.br:~/knn-openmp/$RESULTS_DIR/ ./resultados/"
echo ""
echo "Finalizado em: $(date)"
