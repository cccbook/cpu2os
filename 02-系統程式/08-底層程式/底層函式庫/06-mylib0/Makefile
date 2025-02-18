CC = gcc -w 
OBJS = hello input printftest scanftest test
all: $(OBJS)

hello: hello.c mylib.c
	$(CC)  $^ -o $@

input: input.c mylib.c
	$(CC)  $^ -o $@

printftest: printftest.c mylib.c
	$(CC)  $^ -o $@

scanftest: scanftest.c mylib.c
	$(CC)  $^ -o $@

test: test.c mylib.c
	$(CC)  $^ -o $@

clean:
	rm *.exe $(OBJS)
