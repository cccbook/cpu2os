riscv64-unknown-elf-gcc -S -march=rv32g -mabi=ilp32 -static -mcmodel=medany \
  -fvisibility=hidden -nostdlib -nostartfiles -Thello.ld hello.c -o hello.s
