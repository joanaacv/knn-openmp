CFLAGS = -O2 -Wall
LDFLAGS = -lm
BUILDDIR = build

# Detectar sistema operacional para flags OpenMP
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    CC := $(shell which gcc-16 2>/dev/null)
    ifeq ($(CC),)
        CC := $(shell which gcc-14 2>/dev/null)
    endif
    ifeq ($(CC),)
        CC := $(shell which gcc-13 2>/dev/null)
    endif
    ifneq ($(CC),)
        OMPFLAGS = -fopenmp
    else
        CC = clang
        OMPFLAGS = -Xclang -fopenmp
        LDFLAGS += -L/opt/homebrew/opt/libomp/lib -lomp
        CFLAGS += -I/opt/homebrew/opt/libomp/include
    endif
else
    # Linux (PCAD hype, etc.): gcc padrao com OpenMP
    CC = gcc
    OMPFLAGS = -fopenmp
endif

# ===========================================================================
# Binarios (todos dentro de build/)
# ===========================================================================
SEQ = $(BUILDDIR)/knn_seq

PAR_STATIC_SIMD    = $(BUILDDIR)/knn_par_static_simd
PAR_STATIC_NOSIMD  = $(BUILDDIR)/knn_par_static_nosimd
PAR_DYNAMIC_SIMD   = $(BUILDDIR)/knn_par_dynamic_simd
PAR_DYNAMIC_NOSIMD = $(BUILDDIR)/knn_par_dynamic_nosimd
PAR_GUIDED_SIMD    = $(BUILDDIR)/knn_par_guided_simd
PAR_GUIDED_NOSIMD  = $(BUILDDIR)/knn_par_guided_nosimd

ALL_PAR = $(PAR_STATIC_SIMD) $(PAR_STATIC_NOSIMD) \
          $(PAR_DYNAMIC_SIMD) $(PAR_DYNAMIC_NOSIMD) \
          $(PAR_GUIDED_SIMD) $(PAR_GUIDED_NOSIMD)

ALL = $(SEQ) $(ALL_PAR)

.PHONY: all clean seq par benchmark generate-synthetic benchmark-50k run-all-50k

all: $(ALL)

seq: $(SEQ)

par: $(ALL_PAR)

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

# ---------------------------------------------------------------------------
# Sequencial (baseline)
# ---------------------------------------------------------------------------
$(SEQ): knn_sequencial.c | $(BUILDDIR)
	$(CC) $(CFLAGS) $(OMPFLAGS) -o $@ $< $(LDFLAGS)

# ---------------------------------------------------------------------------
# Variantes paralelas
# SCHEDULE_TYPE: 1=static, 2=dynamic, 3=guided
# USE_SIMD: 1=on, 0=off
# ---------------------------------------------------------------------------
$(PAR_STATIC_SIMD): knn_parallel.c | $(BUILDDIR)
	$(CC) $(CFLAGS) $(OMPFLAGS) -DSCHEDULE_TYPE=1 -DUSE_SIMD=1 -o $@ $< $(LDFLAGS)

$(PAR_STATIC_NOSIMD): knn_parallel.c | $(BUILDDIR)
	$(CC) $(CFLAGS) $(OMPFLAGS) -DSCHEDULE_TYPE=1 -DUSE_SIMD=0 -o $@ $< $(LDFLAGS)

$(PAR_DYNAMIC_SIMD): knn_parallel.c | $(BUILDDIR)
	$(CC) $(CFLAGS) $(OMPFLAGS) -DSCHEDULE_TYPE=2 -DUSE_SIMD=1 -o $@ $< $(LDFLAGS)

$(PAR_DYNAMIC_NOSIMD): knn_parallel.c | $(BUILDDIR)
	$(CC) $(CFLAGS) $(OMPFLAGS) -DSCHEDULE_TYPE=2 -DUSE_SIMD=0 -o $@ $< $(LDFLAGS)

$(PAR_GUIDED_SIMD): knn_parallel.c | $(BUILDDIR)
	$(CC) $(CFLAGS) $(OMPFLAGS) -DSCHEDULE_TYPE=3 -DUSE_SIMD=1 -o $@ $< $(LDFLAGS)

$(PAR_GUIDED_NOSIMD): knn_parallel.c | $(BUILDDIR)
	$(CC) $(CFLAGS) $(OMPFLAGS) -DSCHEDULE_TYPE=3 -DUSE_SIMD=0 -o $@ $< $(LDFLAGS)

clean:
	rm -rf $(BUILDDIR)

# ---------------------------------------------------------------------------
# Atalhos de execucao (uso: make run-seq N=200)
# ---------------------------------------------------------------------------
DATASET = dados_50k.csv
K = 5
N = 0

run-seq: $(SEQ)
	./$(SEQ) $(DATASET) $(K) $(N)

run-dynamic-simd: $(PAR_DYNAMIC_SIMD)
	./$(PAR_DYNAMIC_SIMD) $(DATASET) $(K) $(N)

run-static-simd: $(PAR_STATIC_SIMD)
	./$(PAR_STATIC_SIMD) $(DATASET) $(K) $(N)

run-guided-simd: $(PAR_GUIDED_SIMD)
	./$(PAR_GUIDED_SIMD) $(DATASET) $(K) $(N)

# Benchmark completo
benchmark: $(ALL)
	./benchmark.sh

generate-synthetic: $(BUILDDIR)/generate_synthetic

benchmark-50k: $(ALL) $(BUILDDIR)/generate_synthetic
	./run_benchmark_50k.sh

run-all-50k: $(ALL) $(BUILDDIR)/generate_synthetic
	./run_all_50k.sh

$(BUILDDIR)/generate_synthetic: generate_synthetic.c | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ $<
