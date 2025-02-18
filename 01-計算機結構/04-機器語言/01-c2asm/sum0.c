#include <stdio.h>

int sum(int n) {
    int s=0;
    for (int i=1; i<=n; i++) {
        s += i;
    }
    return s;
}

int main() {
    printf("sum(10)=%d\n", sum(10));
}

