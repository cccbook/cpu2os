#include <stdio.h>

// sum(n) = 1+2+...+n
int sum(int n) {
  int s;
  int i;
  s=0;
  i=1;
  printf("i=%d s=%d\n", i, s);
  while (i <= n) {
    printf("i=%d s=%d\n", i, s);
    s = s + i;
    i ++;
  }
  return s;
}

int main() {
  printf("sum(10)=%d\n", sum(10));
}
