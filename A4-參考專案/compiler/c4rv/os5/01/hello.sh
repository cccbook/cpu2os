riscv64-unknown-elf-gcc -march=rv32g -mabi=ilp32 -static -mcmodel=medany \
  -fvisibility=hidden -nostdlib -nostartfiles -Thello.ld hello.s -o hello
file hello
qemu-system-riscv32 -nographic -machine sifive_u -bios none -kernel hello