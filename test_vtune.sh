#!/bin/bash
# ===========================================================================
# Teste rapido do VTune (7 casos com N=40000)
# Usar ANTES do run_completo.sh para validar que o VTune funciona.
# Resultados ficam em test-vtune/ (descartavel, nao usado na pesquisa).
#
# Uso: ./test_vtune.sh
# ===========================================================================

# Carregar VTune ANTES do set -u (vtune-vars.sh usa variaveis nao definidas)
source /home/intel/oneapi/vtune/2021.1.1/vtune-vars.sh 2>/dev/null || true

set -euo pipefail

DATASET_FILE="${DATASET:-dados_50k.csv}"
K=5
N=40000
VTUNE_DIR="test-vtune"
REPORTS_DIR="$VTUNE_DIR/reports"
B="build"

CASES="
knn_seq:1:seq_baseline
knn_par_dynamic_simd:1:dynamic_simd_1T
knn_par_dynamic_simd:8:dynamic_simd_8T
knn_par_dynamic_simd:20:dynamic_simd_20T
knn_par_dynamic_simd:40:dynamic_simd_40T
knn_par_static_simd:8:static_simd_8T
knn_par_dynamic_nosimd:8:dynamic_nosimd_8T
"

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
mkdir -p "$VTUNE_DIR" "$REPORTS_DIR"

echo "============================================"
echo "TESTE VTune (7 casos, N=$N)"
echo "Maquina: $(hostname)"
echo "Data: $(date)"
echo "============================================"
echo ""

FALHAS=0
for ENTRY in $CASES; do
    [ -z "$ENTRY" ] && continue
    BIN=$(echo "$ENTRY" | cut -d: -f1)
    NT=$(echo "$ENTRY" | cut -d: -f2)
    TAG=$(echo "$ENTRY" | cut -d: -f3)

    echo -n "  [$TAG] coleta... "
    if [ "$BIN" = "knn_seq" ]; then
        vtune -collect hotspots -knob sampling-mode=hw -result-dir "$VTUNE_DIR/hotspots-$TAG" \
              -- $B/$BIN $DATASET_FILE $K $N > /dev/null 2>&1 && COLETA_OK=1 || COLETA_OK=0
    else
        vtune -collect hotspots -knob sampling-mode=hw -result-dir "$VTUNE_DIR/hotspots-$TAG" \
              -- env OMP_NUM_THREADS=$NT $B/$BIN $DATASET_FILE $K $N > /dev/null 2>&1 && COLETA_OK=1 || COLETA_OK=0
    fi

    if [ "$COLETA_OK" -eq 1 ]; then
        # Exportar reports automaticamente
        vtune -report hotspots -r "$VTUNE_DIR/hotspots-$TAG" -format=csv > "$REPORTS_DIR/hotspots-${TAG}.csv" 2>&1
        vtune -report summary -r "$VTUNE_DIR/hotspots-$TAG" -format=csv > "$REPORTS_DIR/summary-${TAG}.csv" 2>&1
        echo "OK (reports em $REPORTS_DIR/)"
    else
        echo "FALHOU"
        FALHAS=$((FALHAS+1))
    fi
done

echo ""
if [ "$FALHAS" -eq 0 ]; then
    echo "TODOS OS 7 TESTES PASSARAM. VTune esta funcionando."
    echo ""
    echo "Reports CSV gerados em: $REPORTS_DIR/"
    ls "$REPORTS_DIR"/*.csv 2>/dev/null
    echo ""
    echo "Pode rodar: ./run_completo.sh"
else
    echo "ATENCAO: $FALHAS teste(s) falharam. Verifique o VTune antes de rodar o completo."
fi
echo ""
echo "Para limpar: rm -rf $VTUNE_DIR/"
