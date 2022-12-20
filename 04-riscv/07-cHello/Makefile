CC = riscv64-unknown-elf-gcc
OBJDUMP = riscv64-unknown-elf-objdump
OPTS = -c -march=rv32g -mabi=ilp32 -static -mcmodel=medany -fvisibility=hidden -nostdlib -nostartfiles -I. 

hello.elf: hello.c
	$(CC) $(OPTS) $^ -o $@

str.elf: str.c
	$(CC) $(OPTS) $^ -o $@

dump: 
	$(OBJDUMP) -d str.elf

# qemu-system-riscv32 -nographic -machine sifive_e -bios none -kernel hello.elf
