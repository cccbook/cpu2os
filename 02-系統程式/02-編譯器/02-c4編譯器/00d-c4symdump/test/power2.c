#include <stdio.h>
int power2(int n) {
   int r, i;
   r = 1;
   i = 1;
   while (i<=n) {
      r = r*2;
      i++;
   }
   return r;
}

int main() {
   printf("power2(3)=%d\n", power2(3));
}