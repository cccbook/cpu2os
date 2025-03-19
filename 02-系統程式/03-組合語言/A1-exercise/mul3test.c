#include <stdio.h>

// 最簡單的版本就是 mul3 函數改用組合語言寫
int mul3(int a, int b, int c) {
   return a*b*c;
}

int main() {
    printf("mul3(3,2,5)=%d\n", mul3(3,2,5));
}