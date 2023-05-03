#include "../lib/lib.h"

/*

https://stackoverflow.com/questions/19446888/adding-signed-and-unsigned-int

From the C99 standard, section 6.3.1.8 ("Usual arithmetic conversions"):

if the operand that has unsigned integer type has rank greater or equal 
to the rank of the type of the other operand, then the operand with signed 
integer type is converted to the type of the operand with unsigned integer type.
*/

int main() {
  printf("sgn(0x1FFFBC, 20)=%d\n", SGN(0x1FFFBC, 20));
  /*
  printf("SGN(0x1FFFBC, 20)=%d\n", SGN(0x1FFFBC, 20));
  printf("-(1<<20)=%d\n", (int32_t)(-(1<<20)));
  printf("0x1FFFBC=%d\n", 0x1FFFBC);
  printf("BITS(0x1FFFBC, 0, 19)=%d\n", (uint32_t)BITS(0x1FFFBC, 0, 19));
  printf("0xFFFBC=%d\n", 0xFFFBC);
  printf("(-(1<<20))+BITS(0x1FFFBC, 0, 19)=%d\n",  (int32_t)(-(1<<20))+(int32_t)BITS(0x1FFFBC, 0, 19));
  printf("(int32_t)(-(1<<20))+(int32_t)0xFFFBC=%d\n",  (int32_t)(-(1<<20))+(int32_t)0xFFFBC);
  printf("(int32_t)(-(1<<20))+(uint32_t)0xFFFBC=%d\n",  (int32_t)(-(1<<20))+(uint32_t)0xFFFBC);
  printf("(int32_t)(-(1<<20))+(uint32_t)0xFFFBC=%d\n", (uint32_t) ((int32_t)(-(1<<20))+(uint32_t)0xFFFBC));
*/
}
