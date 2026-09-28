#!/bin/bash
#SBATCH --job-name=knn-vtune
#SBATCH --partition=hype
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# ===========================================================================
# KNN OpenMP - Analise com Intel VTune Profiler no PCAD (particao hype)
#
# Coleta 3 tipos de analise:
#   1. performance-snapshot: visao geral do desempenho
#   2. hotspots: funcoes que consomem mais tempo
#   3. hpc-performance: analise HPC (vetorizacao, threading, memoria)
#
# Executa com e sem VTune para medir overhead do profiler.
# ===========================================================================

echo "============================================"
echo "Job VTune: $SLURM_JOB_NAME (ID: $SLURM_JOB_ID)"
echo "No: $(hostname)"
echo "Data: $(date)"
echo "============================================"
echo ""

# Carregar ambiente VTune
source /home/intel/oneapi/vtune/2021.1.1/vtune-vars.sh
echo "VTune carregado: $(vtune --version 2>&1 | head -1)"
echo ""

# Copiar dados para o scratch
mkdir -p $SCRATCH/knn-vtune
cp $HOME/knn-openmp/KNNAlgorithmDataset.csv $SCRATCH/knn-vtune/
cp $HOME/knn-openmp/knn_sequencial.c $SCRATCH/knn-vtune/
cp $HOME/knn-openmp/knn_parallel.c $SCRATCH/knn-vtune/
cp $HOME/knn-openmp/Makefile $SCRATCH/knn-vtune/
cd $SCRATCH/knn-vtune

# Compilar com simbolos de debug para o VTune (-g)
export CFLAGS="-O2 -Wall -g"
make clean && make all
echo ""

# ---------------------------------------------------------------------------
# Parametros
# ---------------------------------------------------------------------------
DATASET="KNNAlgorithmDataset.csv"
K=5
N=569             # Usar o dataset completo
THREADS=20        # 1 por core fisico (sem hyperthreading)
BIN="knn_par_dynamic_simd"
VTUNE_DIR="$SCRATCH/knn-vtune/vtune-results"

mkdir -p $VTUNE_DIR

export OMP_NUM_THREADS=$THREADS

# ---------------------------------------------------------------------------
# 1. Execucao SEM VTune (referencia de tempo limpo)
# ---------------------------------------------------------------------------
echo "=========================================="
echo "1. Referencia SEM VTune (medir overhead)"
echo "=========================================="

echo "  Sequencial:"
build/knn_seq $DATASET $K $N
echo ""

echo "  Paralelo ($THREADS threads, dynamic, simd=ON):"
build/$BIN $DATASET $K $N
echo ""

# ---------------------------------------------------------------------------
# 2. Performance Snapshot
# ---------------------------------------------------------------------------
echo "=========================================="
echo "2. VTune: performance-snapshot"
echo "=========================================="

echo "  Sequencial:"
vtune -collect performance-snapshot -result-dir $VTUNE_DIR/perf-snapshot-seq \
      -- build/knn_seq $DATASET $K $N
echo ""

echo "  Paralelo:"
vtune -collect performance-snapshot -result-dir $VTUNE_DIR/perf-snapshot-par \
      -- build/$BIN $DATASET $K $N
echo ""

vtune -report summary -result-dir $VTUNE_DIR/perf-snapshot-seq \
      -report-output $VTUNE_DIR/perf-snapshot-seq-summary.txt 2>/dev/null
vtune -report summary -result-dir $VTUNE_DIR/perf-snapshot-par \
      -report-output $VTUNE_DIR/perf-snapshot-par-summary.txt 2>/dev/null

# ---------------------------------------------------------------------------
# 3. Hotspots
# ---------------------------------------------------------------------------
echo "=========================================="
echo "3. VTune: hotspots"
echo "=========================================="

echo "  Sequencial:"
vtune -collect hotspots -result-dir $VTUNE_DIR/hotspots-seq \
      -- build/knn_seq $DATASET $K $N
echo ""

echo "  Paralelo:"
vtune -collect hotspots -result-dir $VTUNE_DIR/hotspots-par \
      -- build/$BIN $DATASET $K $N
echo ""

vtune -report hotspots -result-dir $VTUNE_DIR/hotspots-seq \
      -report-output $VTUNE_DIR/hotspots-seq-report.txt 2>/dev/null
vtune -report hotspots -result-dir $VTUNE_DIR/hotspots-par \
      -report-output $VTUNE_DIR/hotspots-par-report.txt 2>/dev/null

# ---------------------------------------------------------------------------
# 4. HPC Performance
# ---------------------------------------------------------------------------
echo "=========================================="
echo "4. VTune: hpc-performance"
echo "=========================================="

echo "  Paralelo:"
vtune -collect hpc-performance -result-dir $VTUNE_DIR/hpc-par \
      -- build/$BIN $DATASET $K $N
echo ""

vtune -report summary -result-dir $VTUNE_DIR/hpc-par \
      -report-output $VTUNE_DIR/hpc-par-summary.txt 2>/dev/null

# ---------------------------------------------------------------------------
# Copiar resultados
# ---------------------------------------------------------------------------
echo "Copiando resultados do VTune..."
cp -r $VTUNE_DIR $HOME/knn-openmp/
echo ""
echo "Resultados VTune salvos em: $HOME/knn-openmp/vtune-results/"
echo "  - perf-snapshot-seq/par: Performance Snapshot"
echo "  - hotspots-seq/par: Hotspots Analysis"
echo "  - hpc-par: HPC Performance Analysis"
echo ""
echo "Para visualizar no VTune GUI:"
echo "  vtune-gui $HOME/knn-openmp/vtune-results/hotspots-par"
echo ""
echo "Job VTune finalizado em: $(date)"
