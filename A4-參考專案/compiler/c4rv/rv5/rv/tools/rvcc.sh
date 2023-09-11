riscv64-unknown-elf-gcc -march=rv32im -mabi=ilp32 -S -nostdlib -O0 -o $1.s $1.c
riscv64-unknown-elf-gcc -march=rv32im -mabi=ilp32 -Ttext=0x0 -nostdlib -o $1.elf $1.c
riscv64-unknown-elf-objcopy -O binary $1.elf $1.bin