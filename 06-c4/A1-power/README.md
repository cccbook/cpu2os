

// 1: #include <stdio.h>
// 2: 
// 3: int power(int a, int n) {
// 4:     int s, i;
// 5:     s = 1;
  power = e;
  *e++ = ENT; *e++ = 2;
  *e++ = LLA; *e++ = -1;
  ...

  *e++ = SI;

  int *whileBegin;
  whileBegin = e;  
7:     while (i<=n) {
    LLA  -2
    LI  
    PSH 
    LLA  2
    LI  
    LE  
    BZ   0
8:         s = s * a;
    LLA  -1
    PSH 
    LLA  -1
    LI  
    PSH 
    LLA  3
    LI  
    MUL 
    SI  
9:         i = i + 1;
    LLA  -2
    PSH 
    LLA  -2
    LI  
    PSH 
    IMM  1
    ADD 
    SI  
10:     }
11:     return s;
    JMP  whileBegin
    LLA  -1
    LI  
    LEV 
12: }
    LEV 
13: 
14: int main() {
15:     printf("power(2,3)=%d\n", power(2,3));
    ENT  0
    IMM  -1260814320
    PSH 
    IMM  2
    PSH 
    IMM  3
    PSH 
    JSR  -1260548072
    ADJ  2
    PSH 
    PRTF
    ADJ  2
16: }
    LEV 