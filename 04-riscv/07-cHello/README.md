# RISC-V


```
Hero3C@DESKTOP-O093POU MINGW64 /c/ccc/co/riscv/05-cHello (master)
$ ./build.sh

Hero3C@DESKTOP-O093POU MINGW64 /c/ccc/co/riscv/05-cHello (master)      
$ riscv64-unknown-elf-objdump -d hello.elf

hello.elf:     file format elf32-littleriscv


Disassembly of section .text:

00000000 <main>:
   0:   ff010113                addi    sp,sp,-16
   4:   00112623                sw      ra,12(sp)
   8:   00812423                sw      s0,8(sp)
   c:   01010413                addi    s0,sp,16
  10:   00000517                auipc   a0,0x0
  14:   00050513                mv      a0,a0
  18:   00000097                auipc   ra,0x0
  1c:   000080e7                jalr    ra # 18 <main+0x18>
  20:   00000793                li      a5,0
  24:   00078513                mv      a0,a5
  28:   00c12083                lw      ra,12(sp)
  2c:   00812403                lw      s0,8(sp)
  30:   01010113                addi    sp,sp,16
  34:   00008067                ret
```
