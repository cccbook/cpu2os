
1. 先寫一個可以把 assembly dump 的程式

    $ ./c4 -s hello.c
    1: #include <stdio.h>
    2:
    3: int main()
    4: {
    5:   printf("hello, world\n");
        ENT  0
        IMM  5046432
        PSH
        PRTF
        ADJ  1
    6:   return 0;
        IMM  0
        LEV
    7: }
        LEV

雖然上面的可以 dump ，但是沒列出 printf 等函數的位址，所以沒辦法還原

先寫程式列出系統呼叫的位址 ...

2. 從函數呼叫開始做，先支援 hello.c ，再支援 fib.c