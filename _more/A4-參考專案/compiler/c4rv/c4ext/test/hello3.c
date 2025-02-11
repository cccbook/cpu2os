#include <stdio.h>

void hello() {
  int i;
  i=0;
  printf("i=%d\n", i);
  while (i<10) {
    printf("hello! i=%d\n", i);
    i++;
  }
}

int main() {
  hello();
}
