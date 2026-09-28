#include <stdio.h>
#include <stdlib.h>

#define NUM_FEATURES 30

static unsigned int next_u32(unsigned int *state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "Uso: %s <arquivo.csv> <amostras> [seed]\n", argv[0]);
        return 1;
    }
    long n = strtol(argv[2], NULL, 10);
    if (n <= 0 || n > 50000) {
        fprintf(stderr, "Erro: amostras deve estar entre 1 e 50000\n");
        return 1;
    }
    unsigned int state = (argc == 4) ? (unsigned int)strtoul(argv[3], NULL, 10) : 42u;
    FILE *fp = fopen(argv[1], "w");
    if (!fp) { perror("fopen"); return 1; }

    fprintf(fp, "id,diagnosis");
    for (int f = 0; f < NUM_FEATURES; f++) fprintf(fp, ",feature%d", f + 1);
    fputc('\n', fp);

    for (long row = 0; row < n; row++) {
        int label = (int)(row & 1L);
        fprintf(fp, "%ld,%c", row + 1, label ? 'M' : 'B');
        for (int f = 0; f < NUM_FEATURES; f++) {
            double noise = (double)(next_u32(&state) % 10000) / 10000.0;
            double value = (double)label * 10.0 + noise + (double)f * 0.01;
            fprintf(fp, ",%.6f", value);
        }
        fputc('\n', fp);
    }
    fclose(fp);
    printf("Gerado %ld amostras em %s\n", n, argv[1]);
    return 0;
}
