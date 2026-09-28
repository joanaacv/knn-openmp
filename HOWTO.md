# HOWTO: benchmark KNN com 50.000 amostras

Este procedimento gera um dataset sintético uma única vez e mede o KNN usando os tamanhos `5000`, `10000`, `20000` e `40000`, com até 40 threads.

## Requisitos

No nó de execução, verifique:

```bash
gcc --version
make --version
```

Em Linux, o compilador precisa suportar OpenMP (`gcc -fopenmp`). Registre também a topologia:

```bash
hostname
lscpu
lstopo-no-graphics
```

## Compilar

```bash
make clean
make all generate-synthetic
```

O gerador produzido será `build/generate_synthetic`.

## Gerar o dataset uma vez

```bash
./build/generate_synthetic dados_50k.csv 50000 42
```

Isso cria um arquivo persistente com 50.000 amostras. O arquivo não deve ser recriado entre as medições. Se ele já existir, `run_benchmark_50k.sh` o reutiliza.

## Teste rápido

Antes da campanha completa, valide o fluxo com um caso pequeno:

```bash
N_LIST="5000" THREADS_LIST="1" ./run_benchmark_50k.sh
```

Confirme que o programa termina, informa a acurácia e cria `resultados_knn.csv`.

## Campanha completa

Na máquina NUMA, configure a afinidade:

```bash
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export OMP_DISPLAY_AFFINITY=TRUE
export LC_NUMERIC=C
```

Execute:

```bash
./run_benchmark_50k.sh
```

O script usa automaticamente:

```text
dataset: dados_50k.csv
N:       5000 10000 20000 40000
threads: 1 2 4 8 10 16 20 32 40
```

O mesmo fluxo pode ser chamado pelo Makefile:

```bash
make benchmark-50k
```

## Run all com seleção de casos

O script `run_all_50k.sh` usa todos os casos quando executado sem argumentos:

```bash
./run_all_50k.sh
```

Para selecionar tamanhos e threads:

```bash
./run_all_50k.sh "5000 40000" "20 40"
```

Para incluir coletas VTune nos casos selecionados:

```bash
./run_all_50k.sh --vtune "40000" "20 40"
```

O `--vtune` exige que `vtune` esteja no `PATH` e grava os resultados em `vtune-dados-50k/`. Como as coletas são demoradas, use primeiro uma seleção pequena para validar o ambiente.

O resultado principal fica em `resultados_knn.csv`. Cada linha representa uma execução individual, com tamanho, versão, schedule, SIMD, threads, tempo e acurácia. O campo `tempo_seg` mede somente a função KNN; geração, leitura do CSV, alocação dos conjuntos e cálculo da acurácia ficam fora dessa região.

## Usar outro arquivo ou outra matriz

Para usar um dataset já gerado:

```bash
DATASET=outro.csv ./run_benchmark_50k.sh
```

Para uma matriz menor:

```bash
N_LIST="10000 20000" THREADS_LIST="1 10 20 40" \
  DATASET=dados_50k.csv ./run_benchmark_50k.sh
```

O arquivo precisa conter pelo menos o maior `N` solicitado e no máximo 50.000 amostras.

## Afinidade e dois sockets

Com `OMP_PLACES=cores`, uma thread ocupa um núcleo físico; 20 threads usam os 20 núcleos físicos e 40 threads incluem SMT. A saída de `OMP_DISPLAY_AFFINITY=TRUE` deve ser guardada junto com os resultados para confirmar onde as threads foram executadas.

Para diagnósticos adicionais, quando disponível:

```bash
numactl --cpunodebind=0 --membind=0 ./build/knn_par_dynamic_simd dados_50k.csv 5 10000
numactl --cpunodebind=1 --membind=1 ./build/knn_par_dynamic_simd dados_50k.csv 5 10000
numactl --interleave=all ./build/knn_par_dynamic_simd dados_50k.csv 5 10000
```

## VTune

Apos rodar o benchmark com `run_benchmark_50k.sh`, use o script de VTune selecionado que **nao sobrescreve** o `resultados_knn.csv`:

```bash
source /home/intel/oneapi/vtune/2021.1.1/vtune-vars.sh
./run_vtune_selected.sh
```

Esse script coleta performance-snapshot, hotspots e hpc-performance para 7 casos representativos (N=40000) e salva os reports em `vtune-selected/reports/*.csv`. Os tempos de referencia (sem VTune) ficam em `resultados_vtune_benchmark.csv`.

Alternativamente, `run_all_50k.sh --vtune` coleta hotspots para todas as combinacoes, mas **sobrescreve** o `resultados_knn.csv` por rodar o benchmark novamente.

## Cuidados

- Não regenere o CSV entre execuções.
- Não compare resultados sem registrar afinidade e topologia.
- O benchmark completo executa cerca de 220 processos; reserve tempo suficiente.
- `resultados_knn.csv` é sobrescrito no início de cada campanha; copie o arquivo se quiser preservar uma execução anterior.
- Se o arquivo não existir, o script encerra e mostra o comando de geração.
