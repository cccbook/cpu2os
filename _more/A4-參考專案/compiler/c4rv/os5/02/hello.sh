riscv64-unknown-elf-gcc -S -march=rv32g -mabi=ilp32 -static -mcmodel=medany \
  -fvisibility=hidden -nostdlib -nostartfiles hello.c -o hello.s


	riscv64-unknown-elf-gcc -S fib.c
	riscv64-unknown-elf-gcc -Wl,-Ttext=0x0 -nostdlib -o fib fib.s
	riscv64-unknown-elf-objcopy -O binary fib fib.bin
# riscv64-unknown-elf-gcc -march=rv32g -mabi=ilp32 -static -mcmodel=medany \
#   -fvisibility=hidden -nostdlib -nostartfiles -Thello.ld hello.s -o hello
# file hello
# qemu-system-riscv32 -nographic -machine sifive_u -bios none -kernel hello