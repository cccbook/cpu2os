CC = gcc
CFLAGS = -m32 -Wall -std=gnu99
TARGET = c4 vm4 bc4 bc2c # bc2rv
all: $(TARGET)

c4: c4.c bc.c
	$(CC) -D__C4__ $(CFLAGS) -o $@ $^

vm4: vm4.c bc.c c4.c
	$(CC) $(CFLAGS) -o $@ $^

bc4: bc4.c bc.c c4.c
	$(CC) $(CFLAGS) -o $@ $^

bc2c: bc2c.c c4.c bc.c
	$(CC) $(CFLAGS) -o $@ $^

# bc2rv: bc2rv.c c4.c bc.c
#	$(CC) $(CFLAGS) -o $@ $^

clean:
	$(RM) $(TARGET) *.exe *.o
