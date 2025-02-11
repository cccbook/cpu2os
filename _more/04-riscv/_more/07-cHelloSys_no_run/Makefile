CC = riscv64-unknown-elf-gcc
OBJDUMP = riscv64-unknown-elf-objdump
OPTS = 

hello.elf: hello.c
	$(CC) $(OPTS) $^ -o $@

str.elf: str.c
	$(CC) $(OPTS) $^ -o $@

dump: 
	$(OBJDUMP) -d hello.elf

clean:
	rm *.elf
# qemu-system-riscv32 -nographic -machine sifive_e -bios none -kernel hello.elf
