#include <stdio.h>

#define h 0.001

double df(double (*f)(double), double x) {
    return (f(x+h)-f(x))/h;
}

double square(double x) {
    return x*x;
}

double power3(double x) {
    return x*x*x;
}

int main() {
    printf("df(x**2, 3)=%f\n", df(square, 3));
    printf("df(x**3, 3)=%f\n", df(power3, 3));
}