CC := gcc -std=c99 -m32
LIB = cpu.c riscv.c ../lib/lib.c ../lib/st.c ../lib/map.c
TARGET = as5 ob5dump vm5

all: $(TARGET)

as5: as5.c asm.c parser.c $(LIB)
	$(CC) $^ -o $@

ob5dump: ob5dump.c obj.c $(LIB)
	$(CC) $^ -o $@

vm5: vm5.c vm.c obj.c $(LIB)
	$(CC) $^ -o $@

clean:
	rm -f $(TARGET) *.o *.exe
