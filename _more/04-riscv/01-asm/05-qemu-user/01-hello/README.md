

```
root@localhost:~/cpu2os/04-riscv/01-asm/05-qemu-user/01-hello# riscv64-linux-gnu-gcc -o hello hello.s -static
root@localhost:~/cpu2os/04-riscv/01-asm/05-qemu-user/01-hello# qemu-riscv64 ./hello
Hello, world
```