
```
$ ./c4 test/fib.c
f(7)=13
exit(8) cycle = 920

$ ./c4 -s test/fib.c
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
    JSR  1722970136
    ADJ  1
    PSH 
    LLA  2
    LI  
    PSH 
    IMM  2
    SUB 
    PSH 
    JSR  1722970136
    ADJ  1
    ADD 
    LEV 
7: }
    LEV 
8: 
9: int main() {
10:   printf("f(7)=%d\n", f(7));
    ENT  0
    IMM  1722703888
    PSH 
    IMM  7
    PSH 
    JSR  1722970136
    ADJ  1
    PSH 
    PRTF
    ADJ  2
11: }
    LEV 
```