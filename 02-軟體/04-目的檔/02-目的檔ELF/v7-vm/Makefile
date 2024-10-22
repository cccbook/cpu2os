 
CC = gcc
CFLAGS = -O0

RVCC = riscv64-unknown-elf-gcc
RVDUMP = riscv64-unknown-elf-objdump
RVCFLAGS = -march=rv32g -mabi=ilp32 -nostdlib -Wl,--section-start=.text=0x0

BIN = elf32dump vm32

.PHONY: dump_test vm_test objdump_test # 強制要執行的，放在 .PHONY 裡面

all: elf32dump vm32 rv32

elf32dump: elf32dump.c lib.c dasm.c elf32lib.c
	$(CC) $(CFLAGS) -o $@ $^

vm32: vm32.c vm.c riscv.c lib.c dasm.c
	$(CC) $(CFLAGS) -o $@ $^

rv32: prog/test1.c prog/test2.s
	$(RVCC) $(RVCFLAGS) -S prog/test1.c -o prog/test1_rv32.s
	$(RVCC) $(RVCFLAGS) prog/test1.c -o prog/test1_rv32.o
	$(RVCC) $(RVCFLAGS) -c prog/test2.s -o prog/test2_rv32.o

dump_test:
	./elf32dump prog/test1_rv32.o

objdump_test:
	$(RVDUMP) -d prog/test1_rv32.o

vm_test:
	./vm32 prog/test1_rv32.o
	./vm32 prog/test2_rv32.o

clean:
	$(RM) *.o $(BIN) prog/*.o

.PRECIOUS: %.o tests/%.class tests/%-expected.out tests/%-actual.out tests/%-result.out

indent:
	clang-format -i jvm.c
	cloc jvm.c
