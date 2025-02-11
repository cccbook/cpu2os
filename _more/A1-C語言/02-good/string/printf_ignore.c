#include <stdio.h>
// printf 的 %*s 和我們想的不一樣！
// 參考 -- https://stackoverflow.com/questions/39949473/forcing-printf-to-ignore-formatting
int main() {
  char *a="hello xxxx", *b="world";
  // printf("a=%*s b=%s\n", a, b, b);
  printf("a=%.*s b=%s\n", 3, a, b);
}