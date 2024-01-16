

```
$ clang -S -emit-llvm fib.c

$ clang -S fib.ll -o fib.s

$ clang fib.s main.c -o main2

$ ./main2
fib(5)=8
```