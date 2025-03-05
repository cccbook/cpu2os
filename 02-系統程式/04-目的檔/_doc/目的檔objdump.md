# 目的檔

Windows 使用的目的檔格式為 PE/COFF， Linux 使用的目的檔格式為 ELF 。

對於 ELF 格式的詳細描述，請參考 

* [Linux 的 ELF 格式目的檔](elf)

## objdump 反組譯目的檔

C 語言: add.c

```C
int add(int a, int b) {
  int t = a, x=b;
  return a+b;
}
```


將組合語言轉換為目的檔

```
$ gcc -c add.c -o add.o
```

傾印目的檔

```
$ objdump -s add.o

add.o:     file format pe-i386

Contents of section .text:
 0000 5589e58b 55088b45 0c01d05d c3909090  U...U..E...]....
Contents of section .rdata$zzz:
 0000 4743433a 20287464 6d2d3129 20352e31  GCC: (tdm-1) 5.1
 0010 2e300000                             .0..
```

反組譯目的檔

```
$ objdump -d add.o

add.o:     file format pe-i386


Disassembly of section .text:

00000000 <_add>:
   0:   55                      push   %ebp
   1:   89 e5                   mov    %esp,%ebp
   3:   8b 55 08                mov    0x8(%ebp),%edx
   6:   8b 45 0c                mov    0xc(%ebp),%eax
   9:   01 d0                   add    %edx,%eax
   b:   5d                      pop    %ebp
   c:   c3                      ret
   d:   90                      nop
   e:   90                      nop
   f:   90                      nop
```
