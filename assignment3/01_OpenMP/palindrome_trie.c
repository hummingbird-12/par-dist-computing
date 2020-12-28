#include <ctype.h>
#include <omp.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LEN 32

typedef struct _TRIE_NODE {
    struct _TRIE_NODE* child[128];
    bool end;
} trie_node;

trie_node* trie;
char dictionary[(1 << 15)][MAX_LEN];

void add_trie(const char* string) {
    trie_node* cur = trie;

    for (int i = 0; string[i] != '\0'; i++) {
        const int index = string[i];
#pragma omp critical
        {
            if (cur->child[index] == NULL) {
                cur->child[index] = (trie_node*) calloc(1, sizeof(trie_node));
            }
        }
        cur = cur->child[index];
    }
    cur->end = true;
}

bool query_reverse_trie(const char* query) {
    char reversed[MAX_LEN] = {'\0'};
    for (int i = strlen(query) - 1, j = 0; i >= 0; i--, j++) {
        reversed[j] = query[i];
    }

    trie_node* cur = trie;
    for (int i = 0; reversed[i] != '\0'; i++) {
        const int index = reversed[i];
        if (cur->child[index] == NULL) {
            return false;
        }
        cur = cur->child[index];
    }
    return cur->end;
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
    int words_cnt = 0;
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

    trie = (trie_node*) calloc(1, sizeof(trie_node));

    omp_set_num_threads(THREAD_NUM);
#pragma omp parallel for
    for (int i = 0; i < words_cnt; i++) {
        add_trie(dictionary[i]);
    }

#pragma omp parallel for
    for (int i = 0; i < words_cnt; i++) {
        if (query_reverse_trie(dictionary[i])) {
            fprintf(output, "%s\n", dictionary[i]);
        }
    }
    end = omp_get_wtime();

    printf("Configuration: %d threads\tTime: %f\n", THREAD_NUM, end - start);

    fclose(input);
    fclose(output);

    return 0;
}
