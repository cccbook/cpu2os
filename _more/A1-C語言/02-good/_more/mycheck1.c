#include <stdio.h>

#define check(cond, fmt, ...) \
  if (!cond) \
    printf("Assertion (" #cond ") fail!\nFile: %s, Line %d\n", __FILE__, __LINE__);\
    printf(fmt, __VA_ARGS__);

int main() {
    check(1==0, "Error: %d is not %d\n", 1, 0);
}
