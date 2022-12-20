#include <stdio.h>
#include <stdlib.h>

int main() {
  char *line, *token;
  int age;
  token = malloc(50);
  line = malloc(100);
  sscanf("age 50","%s %d", token, &age);
  sprintf(line, "%s=%d\n", token, age);
  printf(line);
}
