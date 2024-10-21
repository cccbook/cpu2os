base) cccimac@cccimacdeiMac run % ./vm2.sh
+ riscv64-unknown-elf-gcc -march=rv32g -mabi=ilp32 -c test2.s -o test2_rv32.o
+ rm ../vm32.o
+ gcc ../vm32.c -o ../vm32.o
+ ../vm32.o test2_rv32.o
0000 00200093 I addi x1, x0, 2
0004 00308113 I addi x2, x1, 3
0008 06202223 S sw x2, 100(x0)
000c 00008067 I jalr x0, x1, 0