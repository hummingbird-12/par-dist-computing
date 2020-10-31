#ifndef _UTILS_H_INCLUDED_
#define _UTILS_H_INCLUDED_

void flip(const pixel* src, pixel* dest, const int n);
void grayscale(const pixel* src, pixel* dest, const int n);
void smooth(const pixel* src, pixel* dest, const int rows,
        const int cols, const int index);

PPM* create_mod_ppm(const PPM* org, const char* fname_mod);

#endif /* _UTILS_H_INCLUDED_ */
