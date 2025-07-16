#include <stdio.h>

int sum(int n) {
    int s=0; // @s, M=0
    int i=1; // @i, M=1
LOOP:        // (LOOP)
    if (i>n) goto END; // @i, D=M, @n, D=D-M, @END, D;JGT
    // while (i<=n) {
        s = s+i; // @s, D=M, @i, D=D+M, @s, M=D
        i = i+1; // @i, M=M+1
    // }
    goto LOOP; // @LOOP, 0;JMP
END:
    return s;
}

int main() {
    printf("sum(10)=%d\n", sum(10));
}

