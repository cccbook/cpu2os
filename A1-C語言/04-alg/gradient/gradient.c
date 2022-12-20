#include <stdio.h>
#include <string.h>

#define step 0.001

double df(double (*f)(double[], int), double x[], int n, int i) {
    double nx[n];
    memcpy(nx, x, sizeof(double)*n);
    nx[i] += step; 
    return (f(nx, n)-f(x, n))/step;
}

void gradient(double (*f)(double[], int), double x[], int n, double gx[]) {
    for (int i=0; i<n; i++) {
        gx[i] = df(f, x, n, i);
    }
}

#define each(a, n, f) \
    for (int i=0; i<n; i++) { \
        f(a[i]); \
    }

#define dprint(x) printf("%f ", x)

double f(double x[], int n) {
    return x[0]*x[0]+x[1]*x[1];
}

#define N 2

int main(int argc, char *argv[]) {
    double x[N] = {1.0, 2.0};
    double gx[N];
    gradient(f, x, N, gx);
    each(gx, N, dprint);
    printf("\n");
}
