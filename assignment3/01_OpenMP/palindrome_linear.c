#include <ctype.h>
#include <omp.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LEN 32

char dictionary[(1 << 15)][MAX_LEN];
int words_cnt;

bool query_string(const char* query) {
    char reversed[MAX_LEN] = {'\0'};
    for (int i = strlen(query) - 1, j = 0; i >= 0; i--, j++) {
        reversed[j] = query[i];
    }

    for (int i = 0; i < words_cnt; i++) {
        if (strcmp(dictionary[i], reversed) == 0) {
            return true;
        }
    }
    return false;
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s THREAD_NUM INPUT_FILE OUTPUT_FILE\n",
                argv[0]);
        return 1;
    }
    const int THREAD_NUM = atoi(argv[1]);
    const char* INPUT_PATH = argv[2];
    const char* OUTPUT_PATH = argv[3];
    FILE *input, *output;
    double start, end;

    if ((input = fopen(INPUT_PATH, "r")) == NULL) {
        fprintf(stderr, "Error while opening file %s\n", INPUT_PATH);
        return 1;
    }
    if ((output = fopen(OUTPUT_PATH, "w")) == NULL) {
        fprintf(stderr, "Error while opening file %s\n", OUTPUT_PATH);
        return 1;
    }

    start = omp_get_wtime();
    while (fgets(dictionary[words_cnt], MAX_LEN, input) != NULL) {
        const int len = strlen(dictionary[words_cnt]);
        if (strcmp(dictionary[words_cnt] + len - 2, "\r\n") == 0) {
            dictionary[words_cnt][len - 2] = '\0';
        }
        if (strlen(dictionary[words_cnt]) > 0) {
            words_cnt++;
        }
    }

#pragma omp parallel for
    for (int i = 0; i < words_cnt; i++) {
        if (query_string(dictionary[i])) {
            fprintf(output, "%s\n", dictionary[i]);
        }
    }
    end = omp_get_wtime();

    printf("Configuration: %d threads\tTime: %f\n", THREAD_NUM, end - start);

    fclose(input);
    fclose(output);

    return 0;
}
