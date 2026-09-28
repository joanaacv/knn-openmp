#!/bin/bash
# ===========================================================================
# Coleta VTune para casos selecionados (N=40000)
# NAO sobrescreve resultados_knn.csv.
#
# Gera:
#   - resultados_sem_vtune.csv         (tempos SEM vtune - execucao limpa)
#   - resultados_com_vtune.csv         (tempos COM vtune - para medir overhead)
#   - vtune-selected/reports/*.csv     (reports hotspots + summary + hpc)
#   - vtune-selected/{perf,hotspots,hpc}-*/  (dados brutos VTune)
#
# Uso: ./run_vtune_selected.sh
# ===========================================================================

# Carregar VTune ANTES do set -u (vtune-vars.sh usa variaveis nao definidas)
source /home/intel/oneapi/vtune/2021.1.1/vtune-vars.sh 2>/dev/null || true

set -euo pipefail

DATASET_FILE="${DATASET:-dados_50k.csv}"
K=5
N=40000
VTUNE_DIR="vtune-selected"
VTUNE_REPORTS="$VTUNE_DIR/reports"
CSV_SEM="resultados_sem_vtune.csv"
CSV_COM="resultados_com_vtune.csv"
B="build"

# Casos selecionados: binario:threads:descricao
CASES="
knn_seq:1:seq_baseline
knn_par_dynamic_simd:1:dynamic_simd_1T
knn_par_dynamic_simd:8:dynamic_simd_8T
knn_par_dynamic_simd:20:dynamic_simd_20T
knn_par_dynamic_simd:40:dynamic_simd_40T
knn_par_static_simd:8:static_simd_8T
knn_par_dynamic_nosimd:8:dynamic_nosimd_8T
"

# Verificacoes
if ! command -v vtune >/dev/null 2>&1; then
    echo "Erro: vtune nao encontrado. Rode: source /home/intel/oneapi/vtune/2021.1.1/vtune-vars.sh" >&2
    exit 1
fi

if [ ! -f "$DATASET_FILE" ]; then
    echo "Erro: dataset '$DATASET_FILE' nao encontrado. Gere com: $B/generate_synthetic $DATASET_FILE 50000 42" >&2
    exit 1
fi

make all generate-synthetic 2>/dev/null

export OMP_PLACES="${OMP_PLACES:-cores}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-spread}"
export LC_NUMERIC="${LC_NUMERIC:-C}"

rm -rf "$VTUNE_DIR"
mkdir -p "$VTUNE_DIR" "$VTUNE_REPORTS"

echo "============================================"
echo "VTune - Coletas selecionadas (N=$N)"
echo "Maquina: $(hostname)"
echo "Data: $(date)"
echo "Dataset: $DATASET_FILE"
echo "============================================"
echo ""

# Headers dos CSVs
echo "caso,binario,threads,N,tempo_seg,acuracia" > "$CSV_SEM"
echo "caso,binario,threads,N,analise,tempo_seg,acuracia" > "$CSV_COM"

for ENTRY in $CASES; do
    [ -z "$ENTRY" ] && continue
    BIN=$(echo "$ENTRY" | cut -d: -f1)
    NT=$(echo "$ENTRY" | cut -d: -f2)
    TAG=$(echo "$ENTRY" | cut -d: -f3)

    echo "=== [$TAG] $BIN ${NT}T N=$N ==="

    # -----------------------------------------------------------------
    # 1. Execucao SEM VTune (tempo limpo)
    # -----------------------------------------------------------------
    echo "  [sem vtune] Executando..."
    if [ "$BIN" = "knn_seq" ]; then
        OUTPUT=$($B/$BIN $DATASET_FILE $K $N 2>&1)
        TEMPO_SEM=$(echo "$OUTPUT" | grep "Tempo Sequencial" | awk '{print $3}')
    else
        OUTPUT=$(OMP_NUM_THREADS=$NT $B/$BIN $DATASET_FILE $K $N 2>&1)
        TEMPO_SEM=$(echo "$OUTPUT" | grep "Tempo Paralelo" | awk -F': ' '{print $2}' | awk '{print $1}')
    fi
    ACURACIA=$(echo "$OUTPUT" | grep "Acuracia" | awk '{print $2}' | tr -d '%')
    echo "  Tempo sem VTune: ${TEMPO_SEM}s  acc=${ACURACIA}%"
    echo "$TAG,$BIN,$NT,$N,$TEMPO_SEM,$ACURACIA" >> "$CSV_SEM"

    # -----------------------------------------------------------------
    # 2. VTune: performance-snapshot
    # -----------------------------------------------------------------
    echo "  [vtune] performance-snapshot..."
    if [ "$BIN" = "knn_seq" ]; then
        OUTPUT_VTUNE=$($B/$BIN $DATASET_FILE $K $N 2>&1)
        vtune -collect performance-snapshot -result-dir "$VTUNE_DIR/perf-$TAG" \
              -- $B/$BIN $DATASET_FILE $K $N > /dev/null 2>&1
    else
        vtune -collect performance-snapshot -result-dir "$VTUNE_DIR/perf-$TAG" \
              -- env OMP_NUM_THREADS=$NT $B/$BIN $DATASET_FILE $K $N > /dev/null 2>&1
    fi
    vtune -report summary -r "$VTUNE_DIR/perf-$TAG" -format=csv > "$VTUNE_REPORTS/perf-summary-${TAG}.csv" 2>&1

    # -----------------------------------------------------------------
    # 3. VTune: hotspots (sampling-mode=hw)
    # -----------------------------------------------------------------
    echo "  [vtune] hotspots..."
    if [ "$BIN" = "knn_seq" ]; then
        vtune -collect hotspots -knob sampling-mode=hw -result-dir "$VTUNE_DIR/hotspots-$TAG" \
              -- $B/$BIN $DATASET_FILE $K $N > /dev/null 2>&1
    else
        vtune -collect hotspots -knob sampling-mode=hw -result-dir "$VTUNE_DIR/hotspots-$TAG" \
              -- env OMP_NUM_THREADS=$NT $B/$BIN $DATASET_FILE $K $N > /dev/null 2>&1
    fi
    vtune -report hotspots -r "$VTUNE_DIR/hotspots-$TAG" -format=csv > "$VTUNE_REPORTS/hotspots-${TAG}.csv" 2>&1
    vtune -report summary -r "$VTUNE_DIR/hotspots-$TAG" -format=csv > "$VTUNE_REPORTS/hotspots-summary-${TAG}.csv" 2>&1

    # Extrair tempo da execucao com vtune (do report summary)
    TEMPO_COM=$(grep "Elapsed Time" "$VTUNE_REPORTS/hotspots-summary-${TAG}.csv" | head -1 | awk -F'\t' '{print $3}')
    echo "  Tempo com VTune: ${TEMPO_COM}s"
    echo "$TAG,$BIN,$NT,$N,hotspots,$TEMPO_COM,$ACURACIA" >> "$CSV_COM"

    # -----------------------------------------------------------------
    # 4. VTune: hpc-performance (so para paralelo)
    # -----------------------------------------------------------------
    if [ "$BIN" != "knn_seq" ]; then
        echo "  [vtune] hpc-performance..."
        vtune -collect hpc-performance -result-dir "$VTUNE_DIR/hpc-$TAG" \
              -- env OMP_NUM_THREADS=$NT $B/$BIN $DATASET_FILE $K $N > /dev/null 2>&1
        vtune -report summary -r "$VTUNE_DIR/hpc-$TAG" -format=csv > "$VTUNE_REPORTS/hpc-summary-${TAG}.csv" 2>&1
    fi

    echo "  Concluido."
    echo ""
done

echo "============================================"
echo "CONCLUIDO!"
echo "============================================"
echo ""
echo "Arquivos gerados (NAO sobrescrevem resultados_knn.csv):"
echo "  - $CSV_SEM  (tempos SEM VTune - execucao limpa)"
echo "  - $CSV_COM  (tempos COM VTune - para medir overhead)"
echo "  - $VTUNE_REPORTS/ ($(ls "$VTUNE_REPORTS"/*.csv 2>/dev/null | wc -l) reports CSV)"
echo ""
echo "Para baixar para o Mac:"
echo "  rsync -avP jacvargas@gppd-hpc.inf.ufrgs.br:~/knn-openmp/$CSV_SEM ./"
echo "  rsync -avP jacvargas@gppd-hpc.inf.ufrgs.br:~/knn-openmp/$CSV_COM ./"
echo "  rsync -avP jacvargas@gppd-hpc.inf.ufrgs.br:~/knn-openmp/$VTUNE_REPORTS/ ./vtune-reports/"
echo ""
echo "Finalizado em: $(date)"
