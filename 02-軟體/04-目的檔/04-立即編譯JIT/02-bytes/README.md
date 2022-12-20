# JIT1

這個程式可以在 MSYS/WSL 當中執行

## MSYS2 - MINGW32

MINGW32

```
Hero3C@DESKTOP-O093POU MINGW32 /c/ccc/course/sp/05-asm/03-jit/01
$ gcc jit1.c -o jit1

Hero3C@DESKTOP-O093POU MINGW32 /c/ccc/course/sp/05-asm/03-jit/01
$ ./jit1
f(0) = 0
f(1) = 0
f(2) = 0
f(3) = 0
f(4) = 0
f(5) = 0
f(6) = 0
f(7) = 0
f(8) = 0
f(9) = 0

```

## WSL

```
$ wsl
wsl> gcc jit1.c -o jit1
wsl> ./jit1
f(0) = 0
f(1) = 1
f(2) = 2
f(3) = 3
f(4) = 4
f(5) = 5
f(6) = 6
f(7) = 7
f(8) = 8
f(9) = 9
```
