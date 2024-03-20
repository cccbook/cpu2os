int range_sum(int a, int b) {
    int s, i;
    s = 0;
    i = a;
    while (i<=b) {
        s = s + i;
        i = i + 1;
    }
    return s;
}

int main() {
  printf("range_sum(3,9)=%d\n", range_sum(3,9));
}
