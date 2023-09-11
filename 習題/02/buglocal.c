#include <stdio.h>

int *square(int n) {
    int x[1];
    x[0] = n*n;
    return x;
}
/*
int square(int n) {
    return n*n;
}
*/
int main() {
    int *x = square(5);
    printf("x=%d\n", *x);
}
