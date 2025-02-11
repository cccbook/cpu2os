# OS5 -- hello

```
$ make
riscv64-unknown-elf-gcc -fno-common -O0 -march=rv32imafd -mabi=ilp32 -T hello.ld -nostartfiles  hello.c startup.c -o hello.elf
riscv64-unknown-elf-objcopy -Obinary hello.elf hello.bin
riscv64-unknown-elf-objdump -S hello.elf > hello.list
```
