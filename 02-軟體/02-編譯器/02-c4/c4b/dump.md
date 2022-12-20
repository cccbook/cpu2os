

```
$ gcc -w c7.c -o c7
$ ./c7 -s test/hello.c
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
code0=6684816 codeSize=96 data0=6946976 dataLen=16 pc=660098
pcode=4245928 pdata=4245936 *pcode=6684816 *pdata=6946976
$ gcc -w dump.c -o dump
$ ./dump
code0=6684816 codeSize=96 data0=6946976 dataLen=16 pc=660098
```