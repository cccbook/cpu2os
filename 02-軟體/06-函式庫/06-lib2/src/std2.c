#include "std2.h"

void puti(int x) {
    printf("%d ", x);
}

void putf(double x) {
    printf("%f ", x);
}

void _putc(c64_t x) {
    printf("%.4f+%.4fi ", creal(x), cimag(x));
}

void sputi(char *str, int x) {
    sprintf(str, "%d ", x);
}

void sputf(char *str, double x) {
    sprintf(str, "%f ", x);
}

void sputc(char *str, c64_t x) {
    sprintf(str, "%.4f+%.4fi ", creal(x), cimag(x));
}

void *_new(int size, char *file, int line) {
   void *p=malloc(size);
   if (!p) {
     ERROR("new0 malloc() fail!")
   }
   return p;
}

bool ok(bool cond, char *exp, char *file, int line) {
    printf("\033[1;32mCheck:%s  \033[0m\n", exp);
    if (cond)
      printf("\033[1;32m  pass!\033[0m\n");
    else
      printf("\033[1;31m  Error: at FILE=%s LINE=%d\033[0m\n", file, line);
    return cond;
}
