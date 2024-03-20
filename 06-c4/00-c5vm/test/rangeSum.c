int range_sum(int a, int b) {
    int s, i;
    s = 0;
    i = a;
    while (i<=b) {
        s += i;
    }
    return s;
}

int main() {
  printf("range_sum(3,9)=%d\n", range_sum(3,9));
}