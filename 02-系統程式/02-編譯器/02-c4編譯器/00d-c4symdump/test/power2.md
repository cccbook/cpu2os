
```sh
$ ./test.sh
# 以下省略...

$ ./c4 -s test/power2.c
1: #include <stdio.h>
2: int power2(int n) {
3:    int r, i;
4:    r = 1;
    ENT  2
    LLA  -1
    PSH 
    IMM  1
    SI  
5:    i = 1;
    LLA  -2
    PSH 
    IMM  1
    SI  
6:    while (i<=n) {
    LLA  -2
    LI  
    PSH 
    LLA  2
    LI  
    LE  
    BZ   0
7:       r = r*2;
    LLA  -1
    PSH 
    LLA  -1
    LI  
    PSH 
    IMM  2
    MUL 
    SI  
8:       i++;
    LLA  -2
    PSH 
    LI  
    PSH 
    IMM  1
    ADD 
    SI  
    PSH 
    IMM  1
    SUB 
9:    }
10:    return r;
    JMP  939819128
    LLA  -1
    LI  
    LEV 
11: }
    LEV 
12: 
13: int main() {
14:    printf("power2(3)=%d\n", power2(3));
    ENT  0
    IMM  940081152
    PSH 
    IMM  3
    PSH 
    JSR  939819016
    ADJ  1
    PSH 
    PRTF
    ADJ  2
```
