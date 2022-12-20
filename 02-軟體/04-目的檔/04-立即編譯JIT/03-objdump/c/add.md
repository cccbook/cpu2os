## add -O3

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

## add -O0

```
wsl> gcc -c add.c -o add
wsl> objdump -d add

add:     file format elf64-x86-64


Disassembly of section .text:

0000000000000000 <add>:
   0:   f3 0f 1e fa             endbr64
   4:   55                      push   %rbp
   5:   48 89 e5                mov    %rsp,%rbp
   8:   89 7d ec                mov    %edi,-0x14(%rbp)
   b:   89 75 e8                mov    %esi,-0x18(%rbp)
   e:   8b 45 ec                mov    -0x14(%rbp),%eax
  11:   89 45 f8                mov    %eax,-0x8(%rbp)
  14:   8b 45 e8                mov    -0x18(%rbp),%eax
  17:   89 45 fc                mov    %eax,-0x4(%rbp)
  1a:   8b 55 ec                mov    -0x14(%rbp),%edx
  1d:   8b 45 e8                mov    -0x18(%rbp),%eax
  20:   01 d0                   add    %edx,%eax
  22:   5d                      pop    %rbp
  23:   c3                      retq
```