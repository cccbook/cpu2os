set -x
riscv64-unknown-elf-gcc -march=rv32g -mabi=ilp32 -c test1.c -o test1.o
riscv64-unknown-elf-objdump -d test1.o
gcc ../rvm32.c -o ../rvm32.o
../rvm32.o test1.o