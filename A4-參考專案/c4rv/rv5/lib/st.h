#ifndef __STAB_H__
#define __STAB_H__

#include <string.h>

typedef struct _St {
  char *text, *end;
  int size;
  // char *textEnd;
} St;

extern void stNew(St *t, char *text, int size);
extern char *stAdd(St *t, char *str);

#endif
