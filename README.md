# KNN - OpenMP

**Paralelizacao do algoritmo K-Nearest Neighbors (KNN) com OpenMP**

---

### Disciplina

- **Disciplina:** Programacao Paralela (INF01008)
- **Instituicao:** Instituto de Informatica - UFRGS
- **Semestre:** 2026/2

### Integrantes

| Nome    |
|---------|
| Joana   |
| Isadora |

## Sobre o Projeto

Este projeto faz parte do trabalho da disciplina de Programacao Paralela e consiste na implementacao paralela, em **OpenMP**, do algoritmo **K-Nearest Neighbors (KNN)**, com o objetivo de reduzir o tempo de execucao da versao sequencial e analisar o comportamento da aplicacao por meio da ferramenta **Intel VTune Profiler**.

## Dataset

O projeto utiliza um **dataset sintetico** gerado pelo programa `generate_synthetic.c`, que produz ate 50.000 amostras no formato CSV com 30 features numericas e 2 classes (`M` = maligno, `B` = benigno). As classes sao separaveis, permitindo validar a corretude do KNN.

O formato do CSV (colunas: id, diagnosis, 30 features) foi baseado no dataset Wisconsin Breast Cancer Diagnostic, que serviu como referencia durante o desenvolvimento do parser e da estrutura de dados.

```bash
# Gerar dataset com 50.000 amostras (feito automaticamente pelos scripts)
./build/generate_synthetic dados_50k.csv 50000 42
```

**Tamanhos usados nos benchmarks:** N = 5.000, 10.000, 20.000, 40.000 (subconjuntos do dataset de 50.000)

## Algoritmo KNN

O K-Nearest Neighbors (KNN) e um algoritmo de classificacao supervisionado. Para cada ponto de teste:

1. Calcula a distancia Euclidiana para todos os pontos de treino
2. Seleciona os K vizinhos mais proximos (selecao parcial - O(K*N))
3. Classifica por votacao majoritaria entre os K vizinhos

## Estrategia de Paralelizacao

O codigo paralelo utiliza diretivas OpenMP conforme vistas nas aulas:

| Diretiva | Uso | Referencia |
|----------|-----|------------|
| `#pragma omp parallel for` | Distribuir iteracoes do loop de teste entre threads | Aula 6 |
| `schedule(static\|dynamic\|guided)` | Politica de escalonamento das iteracoes | Aula 6 |
| `#pragma omp simd reduction(+:sum)` | Vetorizacao SIMD do calculo de distancia | Aulas 9-10 |

**Por que nao precisa de `critical`/`atomic`/`barrier` (Aula 7)?**
Cada iteracao do loop externo e independente: escreve em `predictions[i]` (posicao exclusiva) e usa variaveis locais (automaticamente `private`). Nao ha dependencias de dados entre iteracoes.

### Variantes compiladas

O Makefile gera **6 binarios paralelos** combinando 3 politicas de schedule x 2 opcoes de SIMD:

| Binario | Schedule | SIMD |
|---------|----------|------|
| `knn_par_static_simd` | static | ON |
| `knn_par_static_nosimd` | static | OFF |
| `knn_par_dynamic_simd` | dynamic | ON |
| `knn_par_dynamic_nosimd` | dynamic | OFF |
| `knn_par_guided_simd` | guided | ON |
| `knn_par_guided_nosimd` | guided | OFF |

## Estrutura do Projeto

```
knn-openmp/
├── knn_sequencial.c         # Versao sequencial do KNN (baseline)
├── knn_parallel.c           # Versao paralela com OpenMP (6 variantes)
├── generate_synthetic.c     # Gerador de dataset sintetico (ate 50k amostras)
├── Makefile                 # Compilacao (binarios em build/)
├── benchmark.sh             # Script de benchmark (core)
├── run_benchmark_50k.sh     # Benchmark com dataset sintetico 50k
├── run_all_50k.sh           # Benchmark + VTune opcional (flag --vtune)
├── run_vtune_selected.sh    # VTune para 7 casos selecionados (recomendado)
├── run_all.sh               # [legado] Benchmark + VTune com dataset original
├── knn_hype.slurm           # Script Slurm para submissao via sbatch
├── knn_vtune.slurm          # Script Slurm para VTune via sbatch
├── GUIA_BENCHMARK.md        # Guia detalhado de benchmark NUMA
├── HOWTO.md                 # Passo a passo rapido
├── Trabalho1_PDP.pdf        # Especificacao do trabalho
├── .gitignore
├── README.md
├── build/                   # [gerado] Binarios compilados
├── resultados/              # [gerado] Resultados organizados
├── vtune-dados-50k/         # [gerado] Dados VTune (run_all_50k.sh)
└── vtune-selected/          # [gerado] Dados VTune (run_vtune_selected.sh)
```

## Compilacao e Execucao

### Compilar

```bash
make clean && make all generate-synthetic
```

Isso compila: `knn_seq`, 6 variantes paralelas, e `generate_synthetic` -- todos em `build/`.

### Executar individualmente

**Argumentos:** `[csv_path] [k] [N]`

| Argumento  | Descricao                                  | Padrao                    |
|------------|--------------------------------------------|---------------------------|
| `csv_path` | Caminho para o arquivo CSV                 | `dados_50k.csv`           |
| `k`        | Numero de vizinhos mais proximos           | `5`                       |
| `N`        | Numero de amostras a usar (0 = todas)      | `0`                       |

```bash
# Sequencial com dataset real
build/knn_seq dados_50k.csv 5

# Paralelo com dataset sintetico, 20 threads, 10000 amostras
OMP_NUM_THREADS=20 build/knn_par_dynamic_simd dados_50k.csv 5 10000
```

### Benchmark com dataset sintetico (recomendado)

```bash
# 1. Gerar dataset (uma vez)
./build/generate_synthetic dados_50k.csv 50000 42

# 2. Rodar benchmark
./run_benchmark_50k.sh
```

Testa: N = {5000, 10000, 20000, 40000} x threads = {1, 2, 4, 8, 10, 16, 20, 32, 40} x 6 variantes.

Ou via Makefile:
```bash
make benchmark-50k
```

### Benchmark com VTune

```bash
source /home/intel/oneapi/vtune/2021.1.1/vtune-vars.sh
./run_all_50k.sh --vtune
```

Gera resultados de benchmark + coletas VTune (hotspots) para todas as combinacoes.

## Ambiente de Execucao - PCAD (Maquinas Hype)

| Caracteristica | Valor |
|----------------|-------|
| **CPU**        | 2 x Intel Xeon E5-2650 v3 @ 2.3 GHz (Haswell) |
| **Cores**      | 20 cores fisicos (10 por socket / NUMA node) |
| **Threads**    | 40 threads logicas (SMT2 / Hyperthreading) |
| **RAM**        | 128 GB DDR4 |

### Configuracao NUMA

Os scripts configuram automaticamente a afinidade de threads para a maquina NUMA:

```bash
export OMP_PLACES=cores           # Uma thread por core fisico
export OMP_PROC_BIND=spread       # Distribuir entre os 2 sockets
export LC_NUMERIC=C               # Ponto decimal correto no CSV
```

### Submissao no PCAD

```bash
# Conectar
ssh <usuario>@gppd-hpc.inf.ufrgs.br

# Alocar no interativo
salloc -p hype -J knn-openmp -t 09:00:00

# Conectar no no alocado
ssh hype<N>

# Dentro do tmux (protege contra queda de SSH)
tmux new -s knn
cd ~/knn-openmp
make clean && make all generate-synthetic
./run_benchmark_50k.sh
```

## Testes de Desempenho

Os testes variam:

- **Numero de threads:** 1, 2, 4, 8, 10, 16, 20, 32, 40
- **Tamanho do conjunto de entrada:** 5.000, 10.000, 20.000, 40.000
- **Politica de schedule:** static vs dynamic vs guided
- **SIMD:** habilitado vs desabilitado

Para cada configuracao, calcular:

- **Speedup:** S(p) = T_seq / T_par(p)
- **Eficiencia:** E(p) = S(p) / p

O CSV de resultados (`resultados_knn.csv`) contem as colunas:
`N, num_train, num_test, versao, schedule, simd, threads, tempo_seg, acuracia`

## Analise com Intel VTune Profiler

```bash
source /home/intel/oneapi/vtune/2021.1.1/vtune-vars.sh

# Opcao 1: casos selecionados (recomendado, ~5 min)
./run_vtune_selected.sh

# Opcao 2: junto com benchmark completo
./run_all_50k.sh --vtune

# Opcao 3: coletas individuais (usar sampling-mode=hw para hotspots)
OMP_NUM_THREADS=8 vtune -collect performance-snapshot -result-dir vtune-perf-8T \
  -- build/knn_par_dynamic_simd dados_50k.csv 5 40000
OMP_NUM_THREADS=8 vtune -collect hotspots -knob sampling-mode=hw -result-dir vtune-hotspots-8T \
  -- build/knn_par_dynamic_simd dados_50k.csv 5 40000
OMP_NUM_THREADS=8 vtune -collect hpc-performance -result-dir vtune-hpc-8T \
  -- build/knn_par_dynamic_simd dados_50k.csv 5 40000
```

**Links uteis:**

- [Tutorial de analise de bottlenecks (Intel)](https://www.intel.com/content/www/us/en/docs/vtune-profiler/tutorial-common-bottlenecks-linux/2025-0/overview.html)
- [Interface de linha de comando do VTune](https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/2024-0/command-line-interface.html)
