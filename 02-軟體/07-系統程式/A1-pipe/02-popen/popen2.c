#include <stdio.h>

int main() {
  char buf[10000];
  FILE *fp = popen("ls -all", "r");
  int len = fread(buf, 1, sizeof(buf), fp);
  buf[len] = '\0';
  printf("%s", buf);
  pclose(fp);
}
