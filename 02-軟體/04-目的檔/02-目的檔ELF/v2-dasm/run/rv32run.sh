set -x
riscv64-unknown-elf-gcc -march=rv32g -mabi=ilp32 -c test1.c -o test1_rv32.o
# riscv64-unknown-elf-objdump -d test1_rv32.o
rm ../elf32dump.o
gcc ../elf32dump.c -o ../elf32dump.o
../elf32dump.o test1_rv32.o
