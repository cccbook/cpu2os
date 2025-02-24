
```sh
(env) cccimac@cccimacdeiMac 00d-c4symdump % ./test.sh
+ gcc -w c4.c -o c4
+ ./c4 -s test/fib.c
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
    JSR  1342472200
    ADJ  1
    PSH 
    LLA  2
    LI  
    PSH 
    IMM  2
    SUB 
    PSH 
    JSR  1342472200
    ADJ  1
    ADD 
    LEV 
7: }
    LEV 
8: 
9: int main() {
10:   printf("f(7)=%d\n", f(7));
    ENT  0
    IMM  1342734336
    PSH 
    IMM  7
    PSH 
    JSR  1342472200
    ADJ  1
    PSH 
    PRTF
    ADJ  2
11: }
    LEV 
+ ./c4 -u test/fib.c
sym[0]: char     len=4 tk=134 class=0 type=0 val=0
sym[1]: else     len=4 tk=135 class=0 type=0 val=0
sym[2]: enum     len=4 tk=136 class=0 type=0 val=0
sym[3]: if       len=2 tk=137 class=0 type=0 val=0
sym[4]: int      len=3 tk=138 class=0 type=0 val=0
sym[5]: return   len=6 tk=139 class=0 type=0 val=0
sym[6]: sizeof   len=6 tk=140 class=0 type=0 val=0
sym[7]: while    len=5 tk=141 class=0 type=0 val=0
sym[8]: open     len=4 tk=133 class=130 type=1 val=30
sym[9]: read     len=4 tk=133 class=130 type=1 val=31
sym[10]: close    len=5 tk=133 class=130 type=1 val=32
sym[11]: printf   len=6 tk=133 class=130 type=1 val=33
sym[12]: malloc   len=6 tk=133 class=130 type=1 val=34
sym[13]: free     len=4 tk=133 class=130 type=1 val=35
sym[14]: memset   len=6 tk=133 class=130 type=1 val=36
sym[15]: memcmp   len=6 tk=133 class=130 type=1 val=37
sym[16]: exit     len=4 tk=133 class=130 type=1 val=38
sym[17]: void     len=4 tk=134 class=0 type=0 val=0
sym[18]: main     len=4 tk=133 class=129 type=1 val=671383992
sym[19]: f        len=1 tk=133 class=129 type=1 val=671383560
sym[20]: n        len=1 tk=133 class=0 type=0 val=0
+ ./c4 test/fib.c
f(7)=13
exit(8) cycle = 920
+ ./c4 hello.c
hello, world
exit(0) cycle = 9
+ ./c4 c4.c hello.c
hello, world
exit(0) cycle = 9
exit(0) cycle = 27265
+ ./c4 c4.c c4.c hello.c
hello, world
exit(0) cycle = 9
exit(0) cycle = 27265
exit(0) cycle = 11797515
```
