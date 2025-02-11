#include <stdio.h>

int main() {
  char x = 'a';
  char *p = &x;
  *p='b';
  printf("*p=%c x=%c\n", *p, x);
}