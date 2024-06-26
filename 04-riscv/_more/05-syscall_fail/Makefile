CC = riscv64-unknown-elf-gcc
OBJDUMP = riscv64-unknown-elf-objdump
OPTS = -march=rv32g -mabi=ilp32 -static -mcmodel=medany -fvisibility=hidden -nostdlib -nostartfiles -I. 

riscvHello.elf: riscvHello.s
	$(CC) $(OPTS) $^ -o $@

dump: 
	$(OBJDUMP) -h riscvHello.elf
	$(OBJDUMP) -d riscvHello.elf
	$(OBJDUMP) -s -j .rodata riscvHello.elf

# qemu-system-riscv32 -nographic -machine sifive_e -bios none -kernel hello.elf

clean:
	rm *.elf