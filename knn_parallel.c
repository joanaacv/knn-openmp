#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define NUM_FEATURES 30   // 30 features do dataset Wisconsin Breast Cancer
#define NUM_CLASSES 2     // M (maligno) = 1, B (benigno) = 0
#define DEFAULT_K 5
#define TRAIN_RATIO 0.8   // 80% treino, 20% teste
#define MAX_LINE 4096
#define MAX_SAMPLES 1000

// ---------------------------------------------------------------------------
// Configuracao em tempo de compilacao:
//   -DSCHEDULE_TYPE=1  -> static
//   -DSCHEDULE_TYPE=2  -> dynamic  (padrao)
//   -DSCHEDULE_TYPE=3  -> guided
//   -DUSE_SIMD=1       -> habilita #pragma omp simd (padrao)
//   -DUSE_SIMD=0       -> desabilita SIMD
// ---------------------------------------------------------------------------
#ifndef SCHEDULE_TYPE
#define SCHEDULE_TYPE 2
#endif

#ifndef USE_SIMD
#define USE_SIMD 1
#endif

#if SCHEDULE_TYPE == 1
    #define SCHEDULE_NAME "static"
#elif SCHEDULE_TYPE == 2
    #define SCHEDULE_NAME "dynamic"
#elif SCHEDULE_TYPE == 3
    #define SCHEDULE_NAME "guided"
#else
    #define SCHEDULE_NAME "unknown"
#endif

#if USE_SIMD == 1
    #define SIMD_NAME "ON"
#else
    #define SIMD_NAME "OFF"
#endif

// ---------------------------------------------------------------------------
// Estruturas
// ---------------------------------------------------------------------------
typedef struct {
    double distance;
    int label;
} Neighbor;

typedef struct {
    float features[MAX_SAMPLES][NUM_FEATURES];
    int labels[MAX_SAMPLES];
    int num_samples;
} Dataset;

// ---------------------------------------------------------------------------
// Leitura do dataset CSV (Wisconsin Breast Cancer)
// Formato: id, diagnosis, feature1, feature2, ..., feature30
// ---------------------------------------------------------------------------
int load_dataset(const char *filename, Dataset *dataset) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Erro: nao foi possivel abrir o arquivo '%s'\n", filename);
        return -1;
    }

    char line[MAX_LINE];
    int row = 0;

    if (fgets(line, sizeof(line), fp) == NULL) {
        fprintf(stderr, "Erro: arquivo vazio\n");
        fclose(fp);
        return -1;
    }

    while (fgets(line, sizeof(line), fp) && row < MAX_SAMPLES) {
        char *token;
        int col = 0;

        line[strcspn(line, "\r\n")] = '\0';

        token = strtok(line, ",");
        while (token != NULL) {
            if (col == 0) {
                // Coluna 'id' - ignorar
            } else if (col == 1) {
                dataset->labels[row] = (token[0] == 'M') ? 1 : 0;
            } else if (col >= 2 && col < 2 + NUM_FEATURES) {
                dataset->features[row][col - 2] = (float)atof(token);
            }
            token = strtok(NULL, ",");
            col++;
        }
        row++;
    }

    fclose(fp);
    dataset->num_samples = row;
    return 0;
}

// ---------------------------------------------------------------------------
// Embaralhar indices com Fisher-Yates shuffle (seed fixa para reprodutibilidade)
// ---------------------------------------------------------------------------
void shuffle_indices(int *indices, int n, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < n; i++) indices[i] = i;
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }
}

// ---------------------------------------------------------------------------
// Calculo de distancia Euclidiana
// Com ou sem vetorizacao SIMD conforme macro USE_SIMD (Aulas 9 e 10)
// ---------------------------------------------------------------------------
double calculate_distance(float *a, float *b) {
    double sum = 0.0;
#if USE_SIMD == 1
    #pragma omp simd reduction(+:sum)
#endif
    for (int i = 0; i < NUM_FEATURES; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

// ---------------------------------------------------------------------------
// Algoritmo KNN paralelo com OpenMP
//
// Diretivas utilizadas (conforme conteudo das Aulas 5-10):
//   - parallel for: worksharing (Aula 6)
//   - schedule(static|dynamic|guided): politica de escalonamento (Aula 6)
//   - simd reduction: vetorizacao SIMD (Aulas 9 e 10)
//
// Cada iteracao do loop externo e independente (sem dependencias de dados),
// portanto nao necessita de critical/atomic/barrier (Aula 7).
// ---------------------------------------------------------------------------
void knn_parallel(float *train_data, int *train_labels, int num_train,
                  float *test_data, int *predictions, int num_test, int k) {

#if SCHEDULE_TYPE == 1
    #pragma omp parallel for schedule(static)
#elif SCHEDULE_TYPE == 2
    #pragma omp parallel for schedule(dynamic)
#elif SCHEDULE_TYPE == 3
    #pragma omp parallel for schedule(guided)
#endif
    for (int i = 0; i < num_test; i++) {
        Neighbor *neighbors = (Neighbor *)malloc(num_train * sizeof(Neighbor));

        float *test_point = test_data + i * NUM_FEATURES;

        for (int j = 0; j < num_train; j++) {
            neighbors[j].distance = calculate_distance(test_point,
                                                       train_data + j * NUM_FEATURES);
            neighbors[j].label = train_labels[j];
        }

        // Selecao parcial: encontrar os K vizinhos mais proximos (O(K*N))
        for (int m = 0; m < k; m++) {
            int min_idx = m;
            for (int n = m + 1; n < num_train; n++) {
                if (neighbors[n].distance < neighbors[min_idx].distance)
                    min_idx = n;
            }
            Neighbor temp = neighbors[m];
            neighbors[m] = neighbors[min_idx];
            neighbors[min_idx] = temp;
        }

        // Votacao majoritaria entre os K vizinhos
        int class_counts[NUM_CLASSES] = {0};
        for (int m = 0; m < k; m++)
            class_counts[neighbors[m].label]++;

        int best_class = 0, max_votes = -1;
        for (int c = 0; c < NUM_CLASSES; c++) {
            if (class_counts[c] > max_votes) {
                max_votes = class_counts[c];
                best_class = c;
            }
        }
        // Escrita na posicao exclusiva 'i' sem necessidade de critical/atomic (Aula 7)
        predictions[i] = best_class;
        free(neighbors);
    }
}

// ---------------------------------------------------------------------------
// main
// Argumentos: [csv_path] [k] [N]
//   csv_path: caminho para o CSV (padrao: KNNAlgorithmDataset.csv)
//   k:        numero de vizinhos (padrao: 5)
//   N:        numero de amostras a usar do dataset (padrao: todas)
// ---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    const char *csv_path = "KNNAlgorithmDataset.csv";
    int k = DEFAULT_K;
    int n_samples = 0; // 0 = usar todas

    if (argc >= 2) csv_path = argv[1];
    if (argc >= 3) k = atoi(argv[2]);
    if (argc >= 4) n_samples = atoi(argv[3]);

    if (k <= 0) { fprintf(stderr, "Erro: K deve ser > 0\n"); return 1; }

    // Carregar dataset
    Dataset dataset;
    if (load_dataset(csv_path, &dataset) != 0) return 1;

    // Se N foi especificado, limitar o numero de amostras
    int total_samples = dataset.num_samples;
    if (n_samples > 0 && n_samples < total_samples)
        total_samples = n_samples;

    // Capturar numero de threads
    int num_threads;
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }

    printf("=== KNN Paralelo com OpenMP ===\n");
    printf("Threads: %d | Schedule: %s | SIMD: %s\n", num_threads, SCHEDULE_NAME, SIMD_NAME);
    printf("Dataset: %d amostras (de %d disponiveis), %d features, %d classes (M/B)\n",
           total_samples, dataset.num_samples, NUM_FEATURES, NUM_CLASSES);

    // Embaralhar e dividir em treino/teste
    int *indices = (int *)malloc(dataset.num_samples * sizeof(int));
    shuffle_indices(indices, dataset.num_samples, 42);

    int num_train = (int)(total_samples * TRAIN_RATIO);
    int num_test  = total_samples - num_train;

    // Alocar conjuntos de treino e teste
    float *train_data   = (float *)malloc(num_train * NUM_FEATURES * sizeof(float));
    int   *train_labels = (int *)malloc(num_train * sizeof(int));
    float *test_data    = (float *)malloc(num_test * NUM_FEATURES * sizeof(float));
    int   *test_labels  = (int *)malloc(num_test * sizeof(int));
    int   *predictions  = (int *)malloc(num_test * sizeof(int));

    for (int i = 0; i < num_train; i++) {
        int idx = indices[i];
        memcpy(train_data + i * NUM_FEATURES,
               dataset.features[idx], NUM_FEATURES * sizeof(float));
        train_labels[i] = dataset.labels[idx];
    }

    for (int i = 0; i < num_test; i++) {
        int idx = indices[num_train + i];
        memcpy(test_data + i * NUM_FEATURES,
               dataset.features[idx], NUM_FEATURES * sizeof(float));
        test_labels[i] = dataset.labels[idx];
    }

    printf("Conjunto de treino: %d amostras\n", num_train);
    printf("Conjunto de teste: %d amostras\n", num_test);
    printf("K = %d\n\n", k);

    // Executar KNN paralelo
    printf("Iniciando KNN Paralelo...\n");
    double start_time = omp_get_wtime();

    knn_parallel(train_data, train_labels, num_train,
                 test_data, predictions, num_test, k);

    double end_time = omp_get_wtime();
    double elapsed = end_time - start_time;

    // Calcular acuracia
    int correct = 0;
    for (int i = 0; i < num_test; i++)
        if (predictions[i] == test_labels[i]) correct++;
    double accuracy = 100.0 * correct / num_test;

    printf("Tempo Paralelo (%d threads, %s, simd=%s): %f segundos\n",
           num_threads, SCHEDULE_NAME, SIMD_NAME, elapsed);
    printf("Acuracia: %.2f%% (%d/%d corretos)\n", accuracy, correct, num_test);

    // Liberar memoria
    free(indices);
    free(train_data);
    free(train_labels);
    free(test_data);
    free(test_labels);
    free(predictions);

    return 0;
}
