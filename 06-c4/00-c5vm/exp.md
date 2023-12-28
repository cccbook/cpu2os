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
    ADD      // a = a + 3 // a = *sp++ +  a;
    SI       // x = a, POP ; // (*(int *)*sp++) = a;
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
    ADJ  4 // ENT 3, 4=3（區域變數）+1 (保存的返回點)
7: }
    LEV // 離開函數
```

注意，LEV 之後由於虛擬機有下列程式，所以最後還是會執行 EXIT 離開。

```
  bp = sp = (int *)((int)sp + poolsz);
  *--sp = EXIT; // call exit if main returns
  *--sp = PSH; t = sp;
  *--sp = argc;
  *--sp = (int)argv;
  *--sp = (int)t;
  return run(pc, bp, sp);
```