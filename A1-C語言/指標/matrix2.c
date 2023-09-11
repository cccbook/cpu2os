void add(double *a, double *b, double *r, int n, int m) {
    for (int i=0; i<n; i++) {
        for (int j=0; j<m; j++) {
            r[i*m+j] = a[i*m+j]+b[i*m+j];
        }
    }
}

int main() {
    double a[2][2]={{1.0,2.0},{3.0,4.0}};
    double b[2][2]={{1.0,2.0},{3.0,4.0}};
    double r[2][2]={{0.0,0.0},{0.0,0.0}};
    add((double*)a,(double*)b,(double*)r,2,2);
}