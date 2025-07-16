# c4-hello

```
(py310) cccimac@cccimacdeiMac 02b-c4hello % ./test.sh
+ gcc -w -g c4.c -o c4
+ ./c4 -s hello.c0
============ symbols ==========
sym[0]:     prin tk=133 class=130 type=1 val=33
============ lex ==============
p=0x10151801e tk=133 sym=    prin
p=0x10151801f tk=40 sym=    ("He
p=0x101518029 tk=34 sym=    "Hel
p=0x10151802a tk=41 sym=     );

p=0x10151802b tk=59 sym=      ;

0: printf("Hello!\n");
============ symbols ==========
sym[0]:     prin tk=133 class=130 type=1 val=33
============ compile ==========
id.name=prin class=130
1: printf("Hello!\n");
    IMM  26574872
    PSH 
    PRTF
    ADJ  1
1> IMM  26574872
2> PSH 
3> PRTF
Hello!
4> ADJ  1
5> EXIT
exit(0) cycle = 5
```
