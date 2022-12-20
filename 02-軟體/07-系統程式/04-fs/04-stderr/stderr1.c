#include <stdio.h>

int main() {
  fprintf(stdout, "Some message!\n");
  fprintf(stderr, "Warning: xxx\n");
  fprintf(stderr, "Error: yyy\n");
}
