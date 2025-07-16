#include <stdio.h>

int main() {
    int R1 = 5; // @5; D=A; @R1; M=D; 
    int R2 = 7; // @7; D=A; @R2; M=D;
    int R0 = R1+R2; // @R1; D=M; @R2; D=D+M; @R0; M=D;
    printf("R0=%d\n", R0);
}
