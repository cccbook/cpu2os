#include <lib2.h>

int main(void) {
  int one=1, two=2;
  OK(one==1);
  OK(two==1);

  printf("one="); PUT(one); BR; 

  double pi=3.14159;
  char str[100]; SPUT(str, pi); printf("pi:str=%s\n", str);

  int *a=NEW(int, 99);
  FREE(a);
  int *b=NEW(int, 999999999);
  FREE(b);
}
