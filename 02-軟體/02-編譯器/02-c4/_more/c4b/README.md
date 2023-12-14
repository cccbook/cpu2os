# C6 -- 550 行的 C 語言編譯器 (從 C4->C5->C6 一路修改過來)

## build

```
$ gcc -w c6.c -o c6
```

## hello.c

```
$ ./c6 test/hello.c
hello, world
exit(0) cycle = 9

$ ./c6 c6.c test/hello.c
hello, world
exit(0) cycle = 9    
exit(0) cycle = 26090

$ ./c6 c6.c c6.c test/hello.c
hello, world
exit(0) cycle = 9
exit(0) cycle = 26090
exit(0) cycle = 10667587

$ ./c6 -s test/hello.c

1: #include <stdio.h>
2: 
3: int main()
4: {
5:   printf("hello, world\n");
 0001     ENT 0
 0003     STR 0
 0005     PSH
 0006     PRTF
 0007     ADJ 1
6:   return 0;
 0009     IMM 0
 000B     LEV
7: }
 000C     LEV
$ ./c6 -d test/hello.c
1> ENT 0
2> STR 0
3> PSH
4> PRTF
hello, world
5> ADJ 1
6> IMM 0
7> LEV
8> PSH
9> EXIT
exit(0) cycle = 9
```

## sum.c

```
$ ./c6 test/sum.c
sum(10)=55
exit(0) cycle = 303

$ ./c6 -s test/sum.c
1: #include <stdio.h>
2: 
3: // sum(n) = 1+2+...+n
4: int sum(int n) {
5:   int s;
6:   int i;
7:   s=0;
 0001     ENT 2
 0003     LEA -1
 0005     PSH
 0006     IMM 0
 0008     SI
8:   i=1;
 0009     LEA -2
 000B     PSH
 000C     IMM 1
 000E     SI
9:   while (i <= n) {
 000F     LEA -2
 0011     LI
 0012     PSH
 0013     LEA 2
 0015     LI
 0016     LE
 0017     BZ  0?
10:     s = s + i;
 0019     LEA -1
 001B     PSH
 001C     LEA -1
 001E     LI
 001F     PSH
 0020     LEA -2
 0022     LI
 0023     ADD
 0024     SI
11:     i ++;
 0025     LEA -2
 0027     PSH
 0028     LI
 0029     PSH
 002A     IMM 1
 002C     ADD
 002D     SI
 002E     PSH
 002F     IMM 1
 0031     SUB
12:   }
13:   return s;
 0032     JMP 000F
 0034     LEA -1
 0036     LI
 0037     LEV
14: }
 0038     LEV
15:
16: int main() {
17:   printf("sum(10)=%d\n", sum(10));
 0039     ENT 0
 003B     STR 0
 003D     PSH
 003E     IMM 10
 0040     PSH
 0041     JSR 0001
 0043     ADJ 1
 0045     PSH
 0046     PRTF
 0047     ADJ 2
18:   return 0;
 0049     IMM 0
 004B     LEV
19: }
 004C     LEV
```

## fib.c

```
$ ./c6 test/fib.c
f(7)=13
exit(8) cycle = 920
$ ./c6 -s test/fib.c
1: #include <stdio.h>
2: 
3: int f(int n) {
4:   if (n<=0) return 0;
 0001     ENT 0
 0003     LEA 2
 0005     LI
 0006     PSH
 0007     IMM 0
 0009     LE
 000A     BZ  0?
 000C     IMM 0
 000E     LEV
5:   if (n==1) return 1;
 000F     LEA 2
 0011     LI
 0012     PSH
 0013     IMM 1
 0015     EQ
 0016     BZ  0?
 0018     IMM 1
 001A     LEV
6:   return f(n-1) + f(n-2);
 001B     LEA 2
 001D     LI
 001E     PSH
 001F     IMM 1
 0021     SUB
 0022     PSH
 0023     JSR 0001
 0025     ADJ 1
 0027     PSH
 0028     LEA 2
 002A     LI
 002B     PSH
 002C     IMM 2
 002E     SUB
 002F     PSH
 0030     JSR 0001
 0032     ADJ 1
 0034     ADD
 0035     LEV
7: }
 0036     LEV
8:
9: int main() {
10:   printf("f(7)=%d\n", f(7));
 0037     ENT 0
 0039     STR 0
 003B     PSH
 003C     IMM 7
 003E     PSH
 003F     JSR 0001
 0041     ADJ 1
 0043     PSH
 0044     PRTF
 0045     ADJ 2
11: }
 0047     LEV


```