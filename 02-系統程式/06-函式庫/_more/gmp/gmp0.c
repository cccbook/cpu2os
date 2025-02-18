#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>

int main(){
  mpz_t  n;
  mpz_init (n);
  mpz_set_ui(n, 2);
  printf("hello %p!\n", n);
  char str[100];
  mpz_get_str(str, 10, n);
  printf("str(n)=%s\n", str);
  // gmp_printf("n=%Zd \n", n);
  return 0;
}
