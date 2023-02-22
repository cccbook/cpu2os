#include <stdio.h>

void swap(int a, int b) {  // 使用指標作為函式參數
    int temp = a;
    a = b;
    b = temp;
}

int main() {
    int x = 10, y = 20;
    printf("x = %d, y = %d\n", x, y);
    swap(x, y);             // 將 x, y 的位址傳入 swap 函式
    printf("x = %d, y = %d\n", x, y);
    return 0;
}