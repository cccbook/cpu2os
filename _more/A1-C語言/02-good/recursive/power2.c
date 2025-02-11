#include <stdio.h>

int power2(int n) {
  if (n==0) return 1;
  return power2(n-1)*2;
}

int main() {
  int p = power2(3);
  printf("p=%d\n", p);
}
