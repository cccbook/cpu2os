CC = riscv64-unknown-elf-gcc
CFLAGS = -Wall -Wextra -std=gnu11 -O2 -ffreestanding -nostdlib -mcmodel=medany
ASFLAGS = -g
LDFLAGS = -T kernel.ld -nostdlib

SRCS = os.c uart.c
OBJS = $(SRCS:.c=.o) entry.o switch.o

TARGET = os.elf

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $^

entry.o: entry.S
	$(CC) $(CFLAGS) -c $< -o $@

switch.o: switch.S
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

run: all
	qemu-system-riscv64 -machine virt -nographic -bios none -kernel $(TARGET)

clean:
	rm -f $(TARGET) $(OBJS)
