#include "st.h"

void stNew(St *t, char *text, int size) {
  t->text = text;
  t->size = size;
  t->end = text;
}

char *stAdd(St *t, char *str) {
  char *tp = t->end;
  strcpy(t->end, str);
  t->end += strlen(str) + 1;
  return tp;
}
