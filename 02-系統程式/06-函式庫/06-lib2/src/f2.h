#pragma once

#include "std2.h"
#include <string.h>

#define EACH(a, n, f) \
    for (int i=0; i<n; i++) { \
        f(a[i]); \
    }

#define MAP(a, n, f, r) \
    for (int i=0; i<n; i++) { \
        r[i] = f(a[i]); \
    }

#define MAP2(a, b, n, f, r) \
    for (int i=0; i<n; i++) { \
        r[i] = f(a[i], b[i]); \
    }

#define REDUCE(a, n, f, r) \
    for (int i=0; i<n; i++) { \
        r = f(r, a[i]); \
    }

#define FILTER(a, n, f, r, m) \
    m=0; \
    for (int i=0; i<n; i++) { \
        if (f(a[i])) r[m++] = a[i]; \
    }

#define ARRAY_EQ(a, b, type, len) (memcmp(a, b, sizeof(type)*len)==0)

#define AT(a, type, i) ((type*)a)[i]

#define LEN(a) (sizeof(a)/sizeof((a)[0]))

#define STR(a, n, r) { \
  int ri = 0; \
  for (int i=0; i<n; i++) { \
      char *p = &r[ri]; \
      SPUT(p, a[i]); \
      ri+=strlen(p); \
  } \
}

// f(x)
#define NOT(x) (!(x))
#define BNOT(x) (~(x))
#define NEG(x) (-(x))

// f(x,y)
#define ADD(x,y) ((x)+(y))
#define SUB(x,y) ((x)-(y))
#define MUL(x,y) ((x)*(y))
#define DIV(x,y) ((x)/(y))
#define MOD(x,y) ((x)%(y))
#define BAND(x,y) ((x)&(y))
#define BOR(x,y) ((x)|(y))
#define BXOR(x,y) ((x)^(y))
#define AND(x,y) ((x)&&(y))
#define OR(x,y) ((x)||(y))
#define SHL(x,y) ((x)<<(y))
#define SHR(x,y) ((x)>>(y))
#define EQ(x,y) ((x)==(y))
#define NE(x,y) ((x)!=(y))
#define LT(x,y) ((x)<(y))
#define LE(x,y) ((x)<=(y))
#define GT(x,y) ((x)>(y))
#define GE(x,y) ((x)>=(y))

#define ALL(a, b, n, f) ({ \
    bool r[n]; \
    MAP2(a, b, n, f, r); \
    bool b = true; \
    REDUCE(r, n, AND, b); \
    b; \
})

#define ANY(a, b, n, f) ({ \
    bool r[n]; \
    MAP2(a, b, n, f, r); \
    bool b = true; \
    REDUCE(r, n, OR, b); \
    b; \
})

#define FIND(a, n, f) ({ \
  int idx=-1; \
  for (int i=0; i<n; i++) { \
    if (f(a[i])) { idx=i; break; } \
  } \
  idx; \
})

