#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ppm.h"

#define BUF_LEN 1024

static void skip_comments(FILE** fp, char** buf);

PPM* read_ppm(const char* fname) {
    FILE* fp = fopen(fname, "rb");

    if (fp == NULL) {
        perror("Error while fopen()");
        exit(1);
    }

    char* buf = (char*) calloc(BUF_LEN, sizeof(char));
    PPM* ppm = (PPM*) calloc(1, sizeof(PPM));
    ppm->fname = (char*) calloc(strlen(fname), sizeof(char));
    strcpy(ppm->fname, fname);

    // Read image information
    skip_comments(&fp, &buf);
    sscanf(buf, "%s", ppm->format);
    skip_comments(&fp, &buf);
    sscanf(buf, "%d %d", &(ppm->cols), &(ppm->rows));
    skip_comments(&fp, &buf);
    sscanf(buf, "%d", &(ppm->max_val));
    free(buf);

    // Check magic number
    if (strcmp(ppm->format, "P6") != 0) {
        perror("Unsupported PPM format");
        exit(1);
    }

    // Read pixels
    ppm->data = (pixel*) calloc(ppm->rows * ppm->cols, sizeof(pixel));
    for (int i = 0; i < ppm->rows; i++) {
        for (int j = 0; j < ppm->cols; j++) {
            fread(&ppm->data[i * ppm->cols + j], sizeof(pixel), 1, fp);
        }
    }

    if (fclose(fp) != 0) {
        perror("Error while fclose()");
        exit(1);
    }

    return ppm;
}

void write_ppm(const PPM* ppm) {
    FILE* fp = fopen(ppm->fname, "wb");

    if (fp == NULL) {
        perror("Error while fopen()");
        exit(1);
    }

    // Write image information
    fprintf(fp, "%s\n", ppm->format);
    fprintf(fp, "%d %d\n", ppm->cols, ppm->rows);
    fprintf(fp, "%d\n", ppm->max_val);

    // Write pixels
    for (int i = 0; i < ppm->rows; i++) {
        for (int j = 0; j < ppm->cols; j++) {
            fwrite(&ppm->data[i * ppm->cols + j], sizeof(pixel), 1, fp);
        }
    }

    if (fclose(fp) != 0) {
        perror("Error while fclose()");
        exit(1);
    }
}

void free_ppm(PPM* ppm) {
    free(ppm->data);
    free(ppm->fname);
    free(ppm);
}

static void skip_comments(FILE** fp, char** buf) {
    // Skip lines starting with '#'
    do {
        fgets(*buf, BUF_LEN, *fp);
    } while ((*buf)[0] == '#');
}

