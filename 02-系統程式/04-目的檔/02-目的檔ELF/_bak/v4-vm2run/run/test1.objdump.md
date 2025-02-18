
+ riscv64-unknown-elf-gcc -march=rv32g -mabi=ilp32 -c test1.c -o test1.o
+ riscv64-unknown-elf-objdump -d test1.o

test1.o:     file format elf32-littleriscv


Disassembly of section .text:

00000000 <add>:
   0:   fe010113                add     sp,sp,-32
   4:   00812e23                sw      s0,28(sp)
   8:   02010413                add     s0,sp,32
   c:   fea42623                sw      a0,-20(s0)
  10:   feb42423                sw      a1,-24(s0)
  14:   fec42703                lw      a4,-20(s0)
  18:   fe842783                lw      a5,-24(s0)
  1c:   00f707b3                add     a5,a4,a5
  20:   00078513                mv      a0,a5
  24:   01c12403                lw      s0,28(sp)
  28:   02010113                add     sp,sp,32
  2c:   00008067                ret

00000030 <sum>:
  30:   fd010113                add     sp,sp,-48
  34:   02812623                sw      s0,44(sp)
  38:   03010413                add     s0,sp,48
  3c:   fca42e23                sw      a0,-36(s0)
  40:   fe042623                sw      zero,-20(s0)
  44:   00100793                li      a5,1
  48:   fef42423                sw      a5,-24(s0)
  4c:   0200006f                j       6c <.L4>

00000050 <.L5>:
  50:   fec42703                lw      a4,-20(s0)
  54:   fe842783                lw      a5,-24(s0)
  58:   00f707b3                add     a5,a4,a5
  5c:   fef42623                sw      a5,-20(s0)
  60:   fe842783                lw      a5,-24(s0)
  64:   00178793                add     a5,a5,1
  68:   fef42423                sw      a5,-24(s0)

0000006c <.L4>:
  6c:   fe842703                lw      a4,-24(s0)
  70:   fdc42783                lw      a5,-36(s0)
  74:   fce7dee3                bge     a5,a4,50 <.L5>
  78:   fec42783                lw      a5,-20(s0)
  7c:   00078513                mv      a0,a5
  80:   02c12403                lw      s0,44(sp)
  84:   03010113                add     sp,sp,48
  88:   00008067                ret

0000008c <main>:
  8c:   fe010113                add     sp,sp,-32
  90:   00112e23                sw      ra,28(sp)
  94:   00812c23                sw      s0,24(sp)
  98:   02010413                add     s0,sp,32
  9c:   00a00513                li      a0,10
  a0:   00000097                auipc   ra,0x0
  a4:   000080e7                jalr    ra # a0 <main+0x14>
  a8:   fea42623                sw      a0,-20(s0)
  ac:   000007b7                lui     a5,0x0
  b0:   00078793                mv      a5,a5
  b4:   fef42423                sw      a5,-24(s0)
  b8:   fe842783                lw      a5,-24(s0)
  bc:   00000897                auipc   a7,0x0
  c0:   00088893                mv      a7,a7
  c4:   00000073                ecall
  c8:   fef42423                sw      a5,-24(s0)
  cc:   00000793                li      a5,0
  d0:   00078513                mv      a0,a5
  d4:   01c12083                lw      ra,28(sp)
  d8:   01812403                lw      s0,24(sp)
  dc:   02010113                add     sp,sp,32
  e0:   00008067                ret