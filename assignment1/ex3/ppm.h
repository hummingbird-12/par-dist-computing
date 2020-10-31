#ifndef _PPM_H_INCLUDED_
#define _PPM_H_INCLUDED_

typedef struct _PIX {
    unsigned char R;
    unsigned char G;
    unsigned char B;
} pixel;

typedef struct _PPM {
    pixel* data;
    char* fname;
    int rows;
    int cols;
    int max_val;
    char format[3];
} PPM;

PPM* read_ppm(const char* fname);
void write_ppm(const PPM* ppm);
void free_ppm(PPM* ppm);

#endif /* _PPM_H_INCLUDED_ */

