// ref: https://www.gnu.org/software/gnu-c-manual/gnu-c-manual.html#The-goto-Statement
#include <stdio.h>

int main() {
  const void * const jumptable[] = {&&L0, &&L1, &&L2};
  for (int i=0; i<3; i++) {
      goto *jumptable[i];

L0: printf("L0\n"); continue;
L1: printf("L1\n"); continue;
L2: printf("L2\n"); continue;
  }
}
