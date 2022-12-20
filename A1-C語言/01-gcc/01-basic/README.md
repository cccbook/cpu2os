# GCC

## 編譯成執行檔 -- 輸出 a.out/a.exe

```
user@DESKTOP-96FRN6B MINGW64 /d/pmedia/陳鍾誠/課程/系統程式/02-gcc/01-basic (master)
$ gcc sum.c

user@DESKTOP-96FRN6B MINGW64 /d/pmedia/陳鍾誠/課程/系統程式/02-gcc/01-basic (master)
$ ./a
sum(10)=55
```

## 編譯成執行檔 -- 使用 -o 參數

```
user@DESKTOP-96FRN6B MINGW64 /d/pmedia/陳鍾誠/課程/系統程式/02-gcc/01-basic (master)
$ gcc sum.c -o sum

user@DESKTOP-96FRN6B MINGW64 /d/pmedia/陳鍾誠/課程/系統程式/02-gcc/01-basic (master)
$ ./sum
sum(10)=55
```
