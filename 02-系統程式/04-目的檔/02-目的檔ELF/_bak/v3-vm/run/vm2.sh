set -x
riscv64-unknown-elf-gcc -march=rv32g -mabi=ilp32 -c test2.s -o test2_rv32.o
# riscv64-unknown-elf-objdump -d test1_rv32.o
rm ../vm32.o
gcc ../vm32.c -o ../vm32.o
../vm32.o test2_rv32.o
