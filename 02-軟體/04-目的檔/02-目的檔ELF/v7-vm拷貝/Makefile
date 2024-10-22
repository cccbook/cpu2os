 
CC = gcc
CFLAGS = 
RVCC = riscv64-unknown-elf-gcc 
RVCFLAGS = -march=rv32g -mabi=ilp32

BIN = elf32dump

.PHONY: run # 強制要執行的，放在 .PHONY 裡面

all: elf32dump rv32

elf32dump: elf32dump.c rv32do.c lib.c dasm.c
	$(CC) $(CFLAGS) -o $@ $<

vm32: vm32.c vm.c rv32do.c riscv.c lib.c dasm.c
	$(CC) $(CFLAGS) -o $@ $<

rv32: prog/test1.c
	$(RVCC) $(RVCFLAGS) -c prog/test1.c -o prog/test1_rv32.o

run: 
	./$(BIN) prog/test1_rv32.o

clean:
	$(RM) *.o $(BIN) prog/*.o

.PRECIOUS: %.o tests/%.class tests/%-expected.out tests/%-actual.out tests/%-result.out

indent:
	clang-format -i jvm.c
	cloc jvm.c
