#include <stdio.h>

int main() {
    int a = 5;
    int *p = &a;
    // int **pp = &p;
    int **pp;
    // *pp = &a;
    pp = &p;
    // **pp = 10;
    *pp = 10;
    // **pp = 10;
    printf("a=%d\n", a);
    printf("p=%p\n", p);
}
