# Guia de benchmark do KNN em máquina NUMA

## Diagnóstico da máquina

O `lstopo-no-graphics` mostra 2 sockets/NUMA nodes, 10 núcleos físicos por socket, 20 núcleos físicos no total e 40 CPUs lógicas com SMT2. O mapeamento é:

```text
socket 0: CPUs 0-9 e irmãos SMT 20-29
socket 1: CPUs 10-19 e irmãos SMT 30-39
```

Há aproximadamente 63 GB por NUMA node e 25 MB de L3 por socket.

Antes dos testes, registre `hostname`, `date`, `lscpu`, `lstopo-no-graphics` e a versão do compilador.

## Afinidade OpenMP

Para distribuir threads pelos núcleos físicos:

```bash
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export OMP_DISPLAY_ENV=TRUE
export OMP_DISPLAY_AFFINITY=TRUE
```

Use 20 threads para avaliar os núcleos físicos e 40 para avaliar também SMT:

```bash
OMP_NUM_THREADS=20 ./build/knn_par_dynamic_simd dados_50k.csv 5 40000
OMP_NUM_THREADS=40 ./build/knn_par_dynamic_simd dados_50k.csv 5 40000
```

## Dados sinteticos

O projeto utiliza um dataset sintetico de 50.000 amostras gerado por `generate_synthetic.c`. O formato segue o padrao do Wisconsin Breast Cancer (id, diagnosis, 30 features), com duas classes separaveis.

O carregador reserva espaço para até 50.000 amostras. Arquivos maiores que esse limite são truncados; para os tamanhos recomendados abaixo, mantenha o arquivo dentro desse limite.

## Geração única dos dados

O gerador é separado do programa medido. Gere o arquivo uma vez, antes do benchmark:

```bash
make generate-synthetic
./build/generate_synthetic dados_50k.csv 50000 42
```

Depois, use o mesmo arquivo em todas as execuções. O terceiro argumento do KNN (`N`) seleciona quantas amostras do arquivo serão usadas, sem regenerar os dados:

```bash
./build/knn_seq dados_50k.csv 5 10000
OMP_NUM_THREADS=40 ./build/knn_par_dynamic_simd dados_50k.csv 5 10000
```

Nesse exemplo, o arquivo tem 50.000 registros, mas o caso usa somente 10.000. A geração ocorre fora do programa KNN e fora da região cronometrada. A leitura e a montagem dos conjuntos também ficam fora do tempo reportado; `tempo_seg` mede somente a função KNN.

O benchmark aceita o arquivo por variável de ambiente e continua usando o terceiro argumento de cada programa como amostra máxima:

```bash
DATASET=dados_50k.csv ./benchmark.sh "5000 10000 20000" "1 10 20 40"
```

Assim, o arquivo é gerado uma vez e cada tamanho da lista mede uma fração prefixada do mesmo arquivo.

## Tamanhos recomendados

O cálculo de distâncias cresce aproximadamente como `N_treino * N_teste * NUM_FEATURES`. Como treino e teste são proporcionais a `N`, o tempo cresce aproximadamente como `N²`.

Para o experimento principal, gere um arquivo com 50.000 amostras e use estes tamanhos:

| Input total | Objetivo |
|---:|---|
| 10.000 | carga inicial |
| 20.000 | carga intermediária |
| 40.000 | carga pesada |
| 50.000 | carga máxima do experimento |

O arquivo pode ser gerado assim:

```bash
make all generate-synthetic
./build/generate_synthetic dados_50k.csv 50000 42
```

O benchmark completo fica:

```bash
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export OMP_DISPLAY_AFFINITY=TRUE
LC_NUMERIC=C DATASET=dados_50k.csv ./benchmark.sh \
  "5000 10000 20000 40000" \
  "1 2 4 8 10 16 20 32 40"
```

O fluxo recomendado está automatizado em `run_benchmark_50k.sh`. Ele compila os binários, gera `dados_50k.csv` somente se o arquivo ainda não existir, configura afinidade e executa a matriz completa:

```bash
./run_benchmark_50k.sh
```

Para trocar o nome do arquivo sem gerar novamente:

```bash
DATASET=outro_dataset.csv ./run_benchmark_50k.sh
```

Para executar pelo Makefile:

```bash
make benchmark-50k
```

O arquivo persistente não é recriado entre execuções. A geração acontece antes do benchmark e não entra em `tempo_seg`.

Escolha o tamanho final observando a duração. O tamanho deve ser o mesmo para todas as variantes comparadas; a execução sequencial pode durar muito mais que a de 40 threads porque o custo cresce aproximadamente com `N²`.

## O que o tempo salvo representa

No `benchmark.sh` original, o programa é executado separadamente para cada combinação de `N`, versão, schedule, SIMD e número de threads. Cada linha de `resultados_knn.csv` contém o tempo de uma única execução do programa:

```text
N,versao,schedule,simd,threads,tempo_seg,acuracia
```

`tempo_seg` não é a soma de todas as execuções do script. O tempo medido fica entre `omp_get_wtime()` antes e depois da função KNN. Ele não inclui leitura do CSV, alocação dos conjuntos, embaralhamento nem cálculo da acurácia.

O tempo total do script é a soma aproximada de todas as combinações e inclui compilação, inicializações e overhead do shell. Ele não deve ser usado como tempo de uma configuração específica.

## Gargalos de leitura e preparação

O carregador CSV é executado novamente em cada processo. Mesmo quando o comando usa `N=10000` em um arquivo com 50000 registros, o programa lê e converte as 50000 linhas antes de selecionar o subconjunto. Esse custo fica fora de `tempo_seg`, mas pode dominar o tempo total do `benchmark.sh`, que inicia muitos processos.

O parser usa `fgets`, `strtok` e `atof`. Isso é adequado para preparar o experimento, mas é mais lento que um formato binário e depende da configuração de locale para interpretar o ponto decimal. Para os arquivos sintéticos, execute com `LC_NUMERIC=C` se o ambiente tiver locale diferente:

```bash
LC_NUMERIC=C DATASET=dados_50k.csv ./benchmark.sh "5000 10000 20000" "1 20 40"
```

Os dados são alocados para até 50.000 amostras em cada execução, mesmo que o arquivo seja menor. Isso reserva cerca de 6 MB para features e rótulos, mas não entra no tempo do KNN. O custo relevante dentro do KNN é a alocação e liberação de `neighbors` para cada amostra de teste; esse custo faz parte da medição e cresce com o tamanho do treino.

Se o objetivo for reduzir o tempo total de preparação, mantenha o CSV persistente e use um arquivo próximo do maior `N` testado. Para eliminar o custo de conversão CSV, uma etapa futura pode gerar um formato binário; isso não é necessário para comparar o tempo da função KNN atual.

## Matriz de testes

Para cada tamanho sintético, compare sequencial e OpenMP com 1, 2, 4, 8, 10, 16, 20, 32 e 40 threads. Teste `static`, `dynamic` e `guided`, com SIMD ligado e desligado. Faça uma execução de aquecimento e pelo menos 5 repetições; use a mediana.

Calcule:

```text
speedup(T) = tempo_sequencial / tempo_paralelo(T)
eficiencia(T) = speedup(T) / T
```

Registre também afinidade, acurácia, frequência, memória e carga do nó.

## Comparações NUMA

Faça pelo menos:

```text
10 threads no socket 0
10 threads no socket 1
20 threads espalhadas pelos dois sockets
40 threads, incluindo SMT
```

Se os dados forem inicializados por uma única thread, muitas páginas podem ficar associadas ao primeiro NUMA node. Para separar os efeitos, use `numactl` quando disponível:

```bash
numactl --cpunodebind=0 --membind=0 ./build/knn_par_dynamic_simd ...
numactl --cpunodebind=1 --membind=1 ./build/knn_par_dynamic_simd ...
numactl --interleave=all ./build/knn_par_dynamic_simd ...
```

## Auditoria do código original

A lógica de cálculo de distância, seleção parcial e votação é equivalente entre as versões. No loop paralelo, cada iteração usa um vetor `neighbors` próprio e escreve somente em `predictions[i]`; não há condição de corrida evidente.

Há, porém, problemas de entrada no código original:

- `k > num_train` acessa `neighbors` fora dos limites;
- `N` pequeno pode gerar treino vazio, acesso inválido e divisão por zero;
- `malloc` não é verificado;
- linhas CSV incompletas não são rejeitadas;
- `MAX_SAMPLES=50000` ainda limita arquivos acima desse tamanho;
- as features não são normalizadas, então colunas de escala maior dominam a distância.

Esses pontos devem ser corrigidos antes de usar o benchmark como resultado final. A acurácia do CSV serve para validar funcionalidade; o input sintético grande serve para medir tempo, speedup, eficiência e comportamento NUMA.

## Cuidados

- Não inclua leitura ou geração dos dados no tempo do KNN.
- Não use uma única execução de microssegundos.
- Mantenha `k`, número de features e divisão treino/teste constantes.
- Evite outros trabalhos no mesmo nó.
- Confirme as afinidades com `OMP_DISPLAY_AFFINITY`.
- Verifique memória e troca com `free -h`, `numastat` e `vmstat`.
