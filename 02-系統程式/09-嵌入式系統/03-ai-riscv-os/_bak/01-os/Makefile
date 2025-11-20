# Makefile
# 設定你的工具鏈前綴
TOOLCHAIN_PREFIX := riscv64-unknown-elf-
# CFLAGS = -nostdlib -fno-builtin -mcmodel=medany -march=rv32ima_zicsr -mabi=ilp32

CC := $(TOOLCHAIN_PREFIX)gcc
AS := $(TOOLCHAIN_PREFIX)as
LD := $(TOOLCHAIN_PREFIX)ld

# RV32IM (Integer, Multiply/Divide)
# CFLAGS := -march=rv32im -mabi=ilp32 -O0 -g -nostdlib -ffreestanding
CFLAGS := -march=rv32ima_zicsr -mabi=ilp32 -O0 -g -nostdlib -ffreestanding -fno-builtin -mcmodel=medany 
ASFLAGS := $(CFLAGS)
LDFLAGS := -T linker.ld -nostdlib -Wl,-Map=os.map

SRCS_C := os.c
SRCS_S := trap.S
OBJS := $(SRCS_C:.c=.o) $(SRCS_S:.S=.o)
TARGET := os.elf

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $(OBJS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.S
	$(CC) $(ASFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET) os.map

# 使用 QEMU 執行
run: $(TARGET)
	@echo "--- Starting QEMU (Press Ctrl+A, then X to quit) ---"
	qemu-system-riscv32 \
		-machine virt \
		-nographic \
		-bios none \
		-kernel $(TARGET)
