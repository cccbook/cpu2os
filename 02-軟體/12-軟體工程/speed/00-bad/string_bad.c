#include <stdio.h>
#include <string.h>

void mystrcpy(char *a, char *b) {
  for (size_t i=0; i<=strlen(b); i++) {
    a[i] = b[i];
  }
}

void mystrcpy2(char *a, char *b) {
  size_t len = strlen(b);
  for (size_t i=0; i<=len; i++) {
    a[i] = b[i];
  }
}

void mystrcpy3(char *a, char *b) {
  char *ap=a, *bp=b;
  while (*bp) {
    *ap++ = *bp++;
  }
  *ap = '\0';
}

int main(void) {
  char x[10], y[] = "hello";
  mystrcpy(x,y);
  printf("x=%s y=%s\n", x, y);
}
