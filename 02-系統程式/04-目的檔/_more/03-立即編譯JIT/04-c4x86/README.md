# c4x86 來源

* https://github.com/EarlGray/c4
    * 指令對應 -- https://github.com/EarlGray/c4/blob/master/JIT.md

## wsl run

```
wsl> sudo apt update
wsl> sudo apt install gcc-multilib
wsl> gcc -w -m32 c4x86.c -o c4x86 -ldl
wsl> ./c4x86 hello.c
hello, world
```

