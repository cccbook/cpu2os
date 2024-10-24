+ gcc ../rvm32.c -o ../rvm32.o
+ ../rvm32.o test1.o
0000 fe010113 I addi x2, x2, -32 # addi 應該沒錯 add sp,sp,-32
0004 00812e23 S sw x8, 28(x2)
0008 02010413 I addi x8, x2, 32
000c fea42623 S sw x10, -20(x8)
0010 feb42423 S sw x11, -24(x8)
0014 fec42703 I slti x14, x8, -20 # lw a4,-20(s0)
0018 fe842783 I slti x15, x8, -24 # lw a5,-24(s0)
001c 00f707b3 R add x15, x14, x15
0020 00078513 I addi x10, x15, 0 # OK! mv a0,a5
0024 01c12403 I slti x8, x2, 28  # lw s0,28(sp)
0028 02010113 I addi x2, x2, 32
002c 00008067 I ret
0030 fd010113 I addi x2, x2, -48
0034 02812623 S sw x8, 44(x2)
0038 03010413 I addi x8, x2, 48
003c fca42e23 S sw x10, -36(x8)
0040 fe042623 S sw x0, -20(x8)
0044 00100793 I addi x15, x0, 1
0048 fef42423 S sw x15, -24(x8)
004c 0200006f J jal x0, 16        # j 6c <.L4>
0050 fec42703 I slti x14, x8, -20 # lw a4,-20(s0)
0054 fe842783 I slti x15, x8, -24
0058 00f707b3 R add x15, x14, x15
005c fef42623 S sw x15, -20(x8)
0060 fe842783 I slti x15, x8, -24
0064 00178793 I addi x15, x15, 1
0068 fef42423 S sw x15, -24(x8)
006c fe842703 I slti x14, x8, -24
0070 fdc42783 I slti x15, x8, -36
0074 fce7dee3 B slti x15, x8, -36
0078 fec42783 I slti x15, x8, -20
007c 00078513 I addi x10, x15, 0
0080 02c12403 I slti x8, x2, 44
0084 03010113 I addi x2, x2, 48
0088 00008067 I ret
008c fe010113 I addi x2, x2, -32
0090 00112e23 S sw x1, 28(x2)
0094 00812c23 S sw x8, 24(x2)
0098 02010413 I addi x8, x2, 32
009c 00a00513 I addi x10, x0, 10
00a0 00000097 U auipc x1, 0x0

00a4 000080e7 I jalr x1, x1, 0
00a8 fea42623 S sw x10, -20(x8)
00ac 000007b7 U lui x15, 0x0

00b0 00078793 I addi x15, x15, 0
00b4 fef42423 S sw x15, -24(x8)
00b8 fe842783 I slti x15, x8, -24
00bc 00000897 U auipc x17, 0x0

00c0 00088893 I addi x17, x17, 0
00c4 00000073 I ecall
00c8 fef42423 S sw x15, -24(x8)
00cc 00000793 I addi x15, x0, 0
00d0 00078513 I addi x10, x15, 0
00d4 01c12083 I slti x1, x2, 28
00d8 01812403 I slti x8, x2, 24
00dc 02010113 I addi x2, x2, 32
00e0 00008067 I ret
entry=008c
