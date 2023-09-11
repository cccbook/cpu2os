CC := gcc -std=c99
LIB = ../lib/map.c
TARGET = map_test

all: $(TARGET)

map_test: map_test.c $(LIB)
	$(CC) $^ -o $@

clean:
	rm -f $(TARGET) *.o *.exe
