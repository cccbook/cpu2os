int temp;

#define PUTS(s) __asm__ ( \
        "li a0, 1\n" \
        "la a1, %1\n" \
        "ecall\n" \
        : "=r"(temp) \
        : "r"(s) \
    )

int sum(int n) {
    int s = 0;
    for (int i=1; i<=n; i++)
        s+=i;
    return s;
}

int main() {
    int s = sum(10);
    PUTS("hello");
}
