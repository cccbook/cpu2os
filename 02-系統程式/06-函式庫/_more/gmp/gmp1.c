#include <gmp.h>
#include <stdio.h>
#include <assert.h>

int main(){
  char inputStr[1024] = "3333333333333333333333";
  mpz_t n;
  int flag;
  char oStr[1024];

  /* 1. Initialize the number */
  mpz_init(n);
  mpz_set_ui(n, 0);

  /* 2. Parse the input string as a base 10 number */
  flag = mpz_set_str(n, inputStr, 10);
  assert (flag == 0); /* If flag is not 0 then the operation failed */

  mpz_get_str(oStr, 10, n);
  printf("str(n)=%s\n", oStr);
  // gmp_printf ("n=%Zd\n", n);

  /* 3. Add one to the number */

  mpz_add_ui(n,n,1); /* n = n + 1 */

  /* 4. Print the result */
  mpz_get_str(oStr, 10, n);
  printf("str(n)=%s\n", oStr);
  // gmp_printf ("n+1=%Zd\n", n);

  /* 5. Square n+1 */
  mpz_mul(n,n,n); /* n = n * n */
  mpz_get_str(oStr, 10, n);
  printf("str(n)=%s\n", oStr);
  // gmp_printf ("(n +1)^2 = %Zd\n", n);

  /* 6. Clean up the mpz_t handles or else we will leak memory */
  mpz_clear(n);
}
