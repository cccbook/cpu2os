# Jit

## JitAdd

Objdump

```
wsl> gcc -O3 -c add.c -o add
wsl> objdump -d add

add:     file format elf64-x86-64


Disassembly of section .text:

0000000000000000 <add>:
   0:   f3 0f 1e fa             endbr64
   4:   8d 04 37                lea    (%rdi,%rsi,1),%eax
   7:   c3                      retq
```

Run

```
wsl> gcc jitAdd.c -o jitAdd
wsl> ./jitAdd
f(2,3) = 5
```

## jitFib

```
wsl> gcc -c -fPIC fib.c -o fib
wsl> objdump -d fib

fib:     file format elf64-x86-64


Disassembly of section .text:

0000000000000000 <fib>:
   0:   f3 0f 1e fa             endbr64
   4:   55                      push   %rbp
   5:   48 89 e5                mov    %rsp,%rbp
   8:   53                      push   %rbx
   9:   48 83 ec 18             sub    $0x18,%rsp
   d:   89 7d ec                mov    %edi,-0x14(%rbp)
  10:   83 7d ec 00             cmpl   $0x0,-0x14(%rbp)
  14:   75 07                   jne    1d <fib+0x1d>
  16:   b8 00 00 00 00          mov    $0x0,%eax
  1b:   eb 2b                   jmp    48 <fib+0x48>
  1d:   83 7d ec 01             cmpl   $0x1,-0x14(%rbp)
  21:   75 07                   jne    2a <fib+0x2a>
  23:   b8 01 00 00 00          mov    $0x1,%eax
  28:   eb 1e                   jmp    48 <fib+0x48>
  2a:   8b 45 ec                mov    -0x14(%rbp),%eax
  2d:   83 e8 01                sub    $0x1,%eax
  30:   89 c7                   mov    %eax,%edi
  32:   e8 c9 ff ff ff          callq  0 <fib>
  37:   89 c3                   mov    %eax,%ebx
  39:   8b 45 ec                mov    -0x14(%rbp),%eax
  3c:   83 e8 02                sub    $0x2,%eax
  3f:   89 c7                   mov    %eax,%edi
  41:   e8 ba ff ff ff          callq  0 <fib>
  46:   01 d8                   add    %ebx,%eax
  48:   48 83 c4 18             add    $0x18,%rsp
  4c:   5b                      pop    %rbx
  4d:   5d                      pop    %rbp
  4e:   c3                      retq
wsl> cd ..

wsl> ./jitFib
f(5) = 13
```

## 更深入 JIT

* https://github.com/spencertipping/jit-tutorial (超讚文章!)

## 更多專案

* Jit 大全 -- https://github.com/wdv4758h/awesome-jit
* https://www.gnu.org/software/lightning/manual/lightning.html
* https://github.com/asmjit/asmjit
* https://www.gnu.org/software/libjit/
* https://gcc.gnu.org/onlinedocs/jit/intro/index.html
* https://github.com/bitfunnel/nativejit/
* https://github.com/v8/v8
* https://github.com/dibyendumajumdar/nanojit