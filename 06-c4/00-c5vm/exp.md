# exp.c

```
$ ./c4 test/exp.c
x=8 y=12 z=20
exit(14) cycle = 41

$ ./c4 -s test/exp.c
1: int main() {
2:     int x,y,z;
3:     x = 3+5;
    ENT  3   // 保留 3 個區域變數空間
    LLA  -1  // 載入 &x 到 a 暫存器
    PSH      // 推 a 到堆疊 (&x)
    IMM  3   // a = 3
    PSH      // PUSH a (3)
    IMM  5   // a = 5
    ADD      // a = a + 3
    SI       // &x = a, POP
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