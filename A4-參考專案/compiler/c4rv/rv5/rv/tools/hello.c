#include <stdio.h>

void cput(char c) {
  *((char*) 0x7FFFF000) = c;
}
/*
void sput(char *str) {
  char *p = str;
  while (*p) cput(*p);
}
*/
int main() {
  cput('h');
  // sput("hello!\n");
}
