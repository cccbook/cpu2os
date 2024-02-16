# 簡介 -- C4 編譯器 

[C4](https://github.com/rswier/c4) 是 [Robert Swierczek](https://github.com/rswier/) 寫的一個小型 C 語言編譯器，全部 527 行的原始碼都在 [c4.c](https://github.com/cccbook/c4/blob/master/c4.c) 裏 。

C4 編譯完成後，會產生一種《堆疊機機器碼》放在記憶體內，然後 [虛擬機](vm) 會立刻執行該機器碼。

以下是 C4 編譯器的用法，C4 可以進行《自我編譯》:

```
gcc -o c4 c4.c  (you may need the -m32 option on 64bit machines)
./c4 hello.c
./c4 -s hello.c

./c4 c4.c hello.c
./c4 c4.c c4.c hello.c
```

C4 在 Windows / Linux / MAC 中都可以執行，使用的完全是標準 C 語言語法！

您也可以閱讀 [更詳細的用法](usage) 以進一步學習 C4，或者直接閱讀我 [加過中文註解的原始碼](https://github.com/cccbook/c4/blob/master/c4.c) 。







