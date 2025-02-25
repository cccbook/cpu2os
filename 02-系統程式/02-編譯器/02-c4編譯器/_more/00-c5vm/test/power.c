#include <stdio.h>

int power(int a, int n) {
    int s, i;
    s = 1;
    i = 1;
    while (i<=n) {
        s = s * a;
        i = i + 1;
    }
    return s;
}

int main() {
    printf("power(2,3)=%d\n", power(2,3));
}
