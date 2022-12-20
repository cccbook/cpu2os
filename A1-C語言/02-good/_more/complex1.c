#include <stdio.h>
#include <complex.h>

int main(){
    int complex a = 1+2I;
    complex double b = 2+I;

    double complex c=a*b;
    printf("%f + %fi\n", creal(c), cimag(c));
}