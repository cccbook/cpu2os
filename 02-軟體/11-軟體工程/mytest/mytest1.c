#include <stdio.h>
#include <stdlib.h>
//#include <assert.h>

#define check(cond) \
  if (!(cond)) { \
     printf("Check %s fail: at FILE=%s LINE=%d\n", #cond, __FILE__, __LINE__); \
  }

int main() {
  int i=1, j=2;
  check(i==1);
  check(j==1);
}
