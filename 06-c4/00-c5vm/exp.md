

```
$ ./c4 -s test/exp.c
1: int main() {
2:     int x,y,z;
3:     x = 3+5;
    ENT  3
    LLA  -1
    PSH 
    IMM  3
    PSH 
    IMM  5
    ADD 
    SI  
4:     y = x+4;
    LLA  -2
    PSH 
    LLA  -1
    LI  
    PSH 
    IMM  4
    ADD 
    SI  
5:     z = x+y;
    LLA  -3
    PSH 
    LLA  -1
    LI  
    PSH 
    LLA  -2
    LI  
    ADD 
    SI  
6:     printf("x=%d y=%d z=%d\n", x, y, z);
    IMM  204820496
    PSH 
    LLA  -1
    LI  
    PSH 
    LLA  -2
    LI  
    PSH 
    LLA  -3
    LI  
    PSH 
    PRTF
    ADJ  4
7: }
    LEV 
```