#include <stdio.h>

int sum(int n) {
    int s=0;
    int i=1;
    while (i<=n) {
    // for (int i=1; i<=n; i++) {
        s += i;
        i ++;
    }
    return s;
}

int main() {
    printf("sum(10)=%d\n", sum(10));
}

