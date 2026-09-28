#!/bin/bash
# ===========================================================================
# Script completo: compila, gera dados, benchmark SEM vtune, VTune COMPLETO.
# Nenhum arquivo e sobrescrito entre etapas.
#
# Gera:
#   resultados_knn_sem_vtune.csv       -- benchmark completo (tempos limpos)
#   resultados_knn_com_vtune.csv       -- mesmas combinacoes, tempos COM VTune
#   vtune-completo/reports/*.csv       -- reports VTune em CSV
#   vtune-completo/{perf,hotspots,hpc}-*/ -- dados brutos VTune
#
# Uso: ./run_completo.sh
# ===========================================================================

# Carregar VTune ANTES do set -u (vtune-vars.sh usa variaveis nao definidas)
source /home/intel/oneapi/vtune/2021.1.1/vtune-vars.sh 2>/dev/null || true

set -euo pipefail

DATASET_FILE="${DATASET:-dados_50k.csv}"
SAMPLES="${SAMPLES:-50000}"
SEED="${SEED:-42}"
K="${K:-5}"
N_LIST="${N_LIST:-5000 10000 20000 40000}"
THREADS_LIST="${THREADS_LIST:-1 2 4 8 10 16 20 32 40}"
B="build"
VTUNE_DIR="vtune-completo"
VTUNE_REPORTS="$VTUNE_DIR/reports"
CSV_SEM="resultados_knn_sem_vtune.csv"
CSV_COM="resultados_knn_com_vtune.csv"

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

# ===================================================================
# ETAPA 1: Compilar
# ===================================================================
echo "[1/4] Compilando..."
make clean && make all generate-synthetic
echo ""

# ===================================================================
# ETAPA 2: Gerar dataset
# ===================================================================
echo "[2/4] Gerando dataset..."
if [ ! -f "$DATASET_FILE" ]; then
    $B/generate_synthetic "$DATASET_FILE" "$SAMPLES" "$SEED"
else
    echo "  Dataset '$DATASET_FILE' ja existe. Reutilizando."
fi
echo ""

# ===================================================================
# ETAPA 3: Benchmark SEM VTune
# ===================================================================
echo "[3/4] Benchmark SEM VTune..."
echo ""

export OMP_PLACES="${OMP_PLACES:-cores}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-spread}"
export OMP_DISPLAY_AFFINITY="${OMP_DISPLAY_AFFINITY:-TRUE}"
export LC_NUMERIC="${LC_NUMERIC:-C}"

DATASET="$DATASET_FILE" RESULTS_FILE="$CSV_SEM" ./benchmark.sh "$N_LIST" "$THREADS_LIST"

echo ""
echo "  Benchmark SEM VTune salvo em: $CSV_SEM"
echo ""

# ===================================================================
# ETAPA 4: VTune COMPLETO (mesmas combinacoes)
# ===================================================================
echo "[4/4] VTune COMPLETO (mesmas combinacoes do benchmark)..."
echo ""

if ! command -v vtune >/dev/null 2>&1; then
    echo "AVISO: vtune nao encontrado. Pulando etapa 4." >&2
    echo "Rode antes: source /home/intel/oneapi/vtune/2021.1.1/vtune-vars.sh"
    echo ""
    echo "============================================"
    echo "PARCIALMENTE CONCLUIDO (sem VTune)"
    echo "============================================"
    exit 0
fi

rm -rf "$VTUNE_DIR"
mkdir -p "$VTUNE_DIR" "$VTUNE_REPORTS"

# Header do CSV com VTune
echo "N,num_train,num_test,versao,schedule,simd,threads,tempo_seg,acuracia" > "$CSV_COM"

for N in $N_LIST; do
    NUM_TRAIN=$((N * 80 / 100))
    NUM_TEST=$((N - NUM_TRAIN))
    echo "  === VTune N=$N ==="

    # Sequencial
    TAG="seq_N${N}"
    echo "    [SEQ] 3 analises..."
    vtune -collect performance-snapshot -result-dir "$VTUNE_DIR/perf-$TAG" \
          -- $B/knn_seq "$DATASET_FILE" $K $N > /dev/null 2>&1
    vtune -report summary -r "$VTUNE_DIR/perf-$TAG" -format=csv > "$VTUNE_REPORTS/perf-summary-${TAG}.csv" 2>&1

    vtune -collect hotspots -knob sampling-mode=hw -result-dir "$VTUNE_DIR/hotspots-$TAG" \
          -- $B/knn_seq "$DATASET_FILE" $K $N > /dev/null 2>&1
    vtune -report hotspots -r "$VTUNE_DIR/hotspots-$TAG" -format=csv > "$VTUNE_REPORTS/hotspots-${TAG}.csv" 2>&1
    vtune -report summary -r "$VTUNE_DIR/hotspots-$TAG" -format=csv > "$VTUNE_REPORTS/hotspots-summary-${TAG}.csv" 2>&1

    # Extrair tempo do hotspots summary
    TEMPO=$(grep "Elapsed Time" "$VTUNE_REPORTS/hotspots-summary-${TAG}.csv" | head -1 | awk -F'\t' '{print $3}')
    ACURACIA=$($B/knn_seq "$DATASET_FILE" $K $N 2>&1 | grep "Acuracia" | awk '{print $2}' | tr -d '%' || echo "N/A")
    echo "$N,$NUM_TRAIN,$NUM_TEST,sequencial,-,-,1,$TEMPO,$ACURACIA" >> "$CSV_COM"

    # Paralelo - todas as variantes x todas as threads
    for ENTRY in $VARIANTS; do
        BIN=$(echo "$ENTRY" | cut -d: -f1)
        SCHED=$(echo "$ENTRY" | cut -d: -f2)
        SIMD_LABEL=$(echo "$ENTRY" | cut -d: -f3)

        for NT in $THREADS_LIST; do
            TAG="par_${SCHED}_simd${SIMD_LABEL}_${NT}T_N${N}"
            echo "    [PAR] $SCHED simd=$SIMD_LABEL ${NT}T..."

            vtune -collect performance-snapshot -result-dir "$VTUNE_DIR/perf-$TAG" \
                  -- env OMP_NUM_THREADS=$NT $B/$BIN "$DATASET_FILE" $K $N > /dev/null 2>&1
            vtune -report summary -r "$VTUNE_DIR/perf-$TAG" -format=csv > "$VTUNE_REPORTS/perf-summary-${TAG}.csv" 2>&1

            vtune -collect hotspots -knob sampling-mode=hw -result-dir "$VTUNE_DIR/hotspots-$TAG" \
                  -- env OMP_NUM_THREADS=$NT $B/$BIN "$DATASET_FILE" $K $N > /dev/null 2>&1
            vtune -report hotspots -r "$VTUNE_DIR/hotspots-$TAG" -format=csv > "$VTUNE_REPORTS/hotspots-${TAG}.csv" 2>&1
            vtune -report summary -r "$VTUNE_DIR/hotspots-$TAG" -format=csv > "$VTUNE_REPORTS/hotspots-summary-${TAG}.csv" 2>&1

            vtune -collect hpc-performance -result-dir "$VTUNE_DIR/hpc-$TAG" \
                  -- env OMP_NUM_THREADS=$NT $B/$BIN "$DATASET_FILE" $K $N > /dev/null 2>&1
            vtune -report summary -r "$VTUNE_DIR/hpc-$TAG" -format=csv > "$VTUNE_REPORTS/hpc-summary-${TAG}.csv" 2>&1

            TEMPO=$(grep "Elapsed Time" "$VTUNE_REPORTS/hotspots-summary-${TAG}.csv" | head -1 | awk -F'\t' '{print $3}')
            echo "$N,$NUM_TRAIN,$NUM_TEST,paralelo,$SCHED,$SIMD_LABEL,$NT,$TEMPO,$ACURACIA" >> "$CSV_COM"
        done
    done
    echo ""
done

N_REPORTS=$(ls "$VTUNE_REPORTS"/*.csv 2>/dev/null | wc -l)

echo ""
echo "============================================"
echo "CONCLUIDO!"
echo "============================================"
echo ""
echo "Arquivos gerados:"
echo "  - $CSV_SEM                (benchmark SEM VTune)"
echo "  - $CSV_COM                (benchmark COM VTune)"
echo "  - $VTUNE_REPORTS/         ($N_REPORTS reports CSV)"
echo ""
echo "Para baixar para o Mac:"
echo "  rsync -avP jacvargas@gppd-hpc.inf.ufrgs.br:~/knn-openmp/$CSV_SEM ./"
echo "  rsync -avP jacvargas@gppd-hpc.inf.ufrgs.br:~/knn-openmp/$CSV_COM ./"
echo "  rsync -avP jacvargas@gppd-hpc.inf.ufrgs.br:~/knn-openmp/$VTUNE_REPORTS/ ./vtune-reports/"
echo ""
echo "Finalizado em: $(date)"
