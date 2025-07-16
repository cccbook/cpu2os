set -x
gcc -m32 -c add.c -o add_x86.o
objdump -d add_x86.o # > rv32objdump.txt
gcc ../elf32dump.c -o ../elf32dump.o
../elf32dump.o add_x86.o # > elf32dump.txt
# cat rv32objdump.txt
# cat elf32dump.txt