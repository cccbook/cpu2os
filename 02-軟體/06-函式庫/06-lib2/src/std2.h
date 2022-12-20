#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <complex.h>
#include <tgmath.h>

#define uint_t unsigned int
#define c32_t complex float
#define c64_t complex double

typedef enum { _I8, _U8, _I16, _U16, _I32, _U32, _I64, _U64, _INT, _UINT, _F32, _F64, _C32, _C64 } _type_t;

// https://stackoverflow.com/questions/2616906/how-do-i-output-coloured-text-to-a-linux-terminal
// https://en.wikipedia.org/wiki/ANSI_escape_code
// https://shengwen1997.gitbooks.io/program_with_c/content/Stringification.html
#define OK(cond, ...) ok(cond, #cond, __FILE__, __LINE__)
#define BR printf("\n")
#define ERROR(...) { printf("\033[1;31mError: "); printf(__VA_ARGS__); printf("\033[0m\n"); }
#define NEW(type, size) _new(sizeof(type)*size, __FILE__, __LINE__)
#define FREE(p) { free(p);p=NULL; }

extern void putf(double x);
extern void puti(int x);
extern void _putc(c64_t x);

extern void sputf(char *str, double x);
extern void sputi(char *str, int x);
extern void sputc(char *str, c64_t x);

extern void *_new(int size, char *file, int line);
extern bool ok(bool cond, char *exp, char *file, int line);

#define PUT(x) _Generic((x), \
              int: puti, \
              c64_t: _putc, \
              double: putf \
)(x)

#define SPUT(str, x) _Generic((x), \
              int: sputi, \
              c64_t: sputc, \
              double: sputf \
)(str, x)
