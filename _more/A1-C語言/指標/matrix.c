double add(double a[][], double b[][], int n, int m) {
    int r[n][m];
    for (int i=0; i<n; i++) {
        for (int j=0; j<m; j++) {
            r[i][j] = a[i][j]+b[i][j];
        }
    }
    return r;
}

int main() {
    double a[2][2]={{1.0,2.0},{3.0,4.0}};
    double b[2][2]={{1.0,2.0},{3.0,4.0}};
    double r[2][2] = add(a,b,2,2);
}