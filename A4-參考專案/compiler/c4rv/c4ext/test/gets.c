#include <stdio.h>
#include <stdlib.h>

int main() {
  char *line;
  line = malloc(100);
  gets(line);
  printf("line=%s\n", line);
}
