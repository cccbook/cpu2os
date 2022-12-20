CROSS_COMPILE ?= riscv64-unknown-elf-
CC := $(CROSS_COMPILE)gcc
CFLAGS = -fno-common -O0 \
	 -march=rv32imafd -mabi=ilp32 \
	 -T hello.ld -nostartfiles \

TARGET = hello.bin
all: $(TARGET)

$(TARGET): hello.c startup.c
	$(CC) $(CFLAGS) $^ -o hello.elf
	$(CROSS_COMPILE)objcopy -Obinary hello.elf hello.bin
	$(CROSS_COMPILE)objdump -S hello.elf > hello.list

qemu: $(TARGET)
	@qemu-system-riscv32 -M ? | grep spike >/dev/null || exit
	@echo "Press Ctrl-A and then X to exit QEMU"
	@echo
	qemu-system-riscv32 -M spike -nographic -kernel hello.bin

clean:
	rm -f *.o *.bin *.elf *.list
