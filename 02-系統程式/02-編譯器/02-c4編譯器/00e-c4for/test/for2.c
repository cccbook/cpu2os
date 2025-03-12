#include <stdio.h>

// sum(n) = 1+2+...+n
int sum(int n) {
  int s, i;
  s = 0;
  printf("s=%d\n", s);
  for (i=1;i <= n; i++) {
    printf("start:i=%d s=%d\n", i, s);
    s = s + i;
    printf("end:i=%d s=%d\n", i, s);
  }
  return s;
}

int main() {
  printf("sum(10)=%d\n", sum(10));
  return 0;
}
