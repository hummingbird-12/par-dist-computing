#include <stdlib.h>
#include <string.h>

#include "ppm.h"

const short dr[] = {-1, -1, 0, 1, 1, 1, 0, -1};
const short dc[] = {0, 1, 1, 1, 0, -1, -1, -1};

static char* modify_fname(const char* src, const char* modifier);

void flip(const pixel* src, pixel* dest, const int n) {
    for (int i = 0; i < n; i++) {
        dest[i] = src[n - i - 1];
    }
}

void grayscale(const pixel* src, pixel* dest, const int n) {
    for (int i = 0; i < n; i++) {
        const int avg = (src[i].R + src[i].B + src[i].G) / 3;
        dest[i].R = dest[i].G = dest[i].B = avg;
    }
}

void smooth(const pixel* src, pixel* dest, const int rows,
        const int cols, const int index) {
    const int r = index / cols;
    const int c = index % cols;
    int R = src[index].R;
    int G = src[index].G;
    int B = src[index].B;
    int cnt = 1;

    for (int i = 0; i < 8; i++) {
        const int nr = r + dr[i];
        const int nc = c + dc[i];
        if (0 <= nr && nr < rows && 0 <= nc && nc < cols) {
            const int ni = nr * cols + nc;
            R += src[ni].R;
            G += src[ni].G;
            B += src[ni].B;
            cnt++;
        }
    }

    dest[index].R = R / cnt;
    dest[index].G = G / cnt;
    dest[index].B = B / cnt;
}

PPM* create_mod_ppm(const PPM* org, const char* fname_mod) {
    const int rows = org->rows;
    const int cols = org->cols;

    PPM* mod = (PPM*) calloc(1, sizeof(PPM));
    mod->rows = rows;
    mod->cols = cols;
    mod->max_val = org->max_val;
    mod->data = (pixel*) calloc(rows * cols, sizeof(pixel));
    mod->fname = modify_fname(org->fname, fname_mod);
    strcpy(mod->format, org->format);

    return mod;
}

/*
 * Returns string made by concatenating:
 * FNAME + MODIFIER + ".ppm"
 */
static char* modify_fname(const char* src, const char* modifier) {
    const int n = strlen(src);
    const int m = strlen(modifier);

    char* result = (char*) calloc(n + m + 1, sizeof(char));
    strncat(result, src, n - 4); // Copy until ".ppm" extension
    strcat(result, modifier);
    strcat(result, ".ppm");

    return result;
}

