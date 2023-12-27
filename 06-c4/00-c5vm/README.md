# C4 編譯器

* https://github.com/ccc-c/c4/wiki

```
ccckmit@asus MINGW64 /d/ccc/ccc112a/cpu2os/01-硬體/06-虛擬機/00-c5vm (master)
$ ./test.sh
++ gcc -w c4.c -o c4
++ ./c4 -s test/fib.c
1: #include <stdio.h>
2:
3: int f(int n) {
4:   if (n<=0) return 0;
    ENT  0
    LLA  2
    LI
    PSH
    IMM  0
    LE
    BZ   0
    IMM  0
    LEV
5:   if (n==1) return 1;
    LLA  2
    LI
    PSH
    IMM  1
    EQ
    BZ   0
    IMM  1
    LEV
6:   return f(n-1) + f(n-2);
    LLA  2
    LI
    PSH
    IMM  1
    SUB
    PSH
    JSR  -1283591064
    ADJ  1
    PSH
    LLA  2
    LI
    PSH
    IMM  2
    SUB
    PSH
    JSR  -1283591064
    ADJ  1
    ADD
    LEV
7: }
    LEV
8:
9: int main() {
10:   printf("f(7)=%d\n", f(7));
    ENT  0
    IMM  -1283328912
    PSH
    IMM  7
    PSH
    JSR  -1283591064
    ADJ  1
    PSH
    PRTF
    ADJ  2
11: }
    LEV
++ ./c4 c4.c hello.c
hello, world
exit(0) cycle = 9
exit(0) cycle = 26036
++ ./c4 c4.c c4.c hello.c
hello, world
exit(0) cycle = 9
exit(0) cycle = 26036
exit(0) cycle = 10271942
```
