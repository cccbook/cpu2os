# riscv 工具鏈

```
$ riscv64-unknown-elf-gcc -Wl,-Ttext=0x0 -nostdlib -o add.elf add.s
d:/ccc/ccc109/sp_more/riscv/freedomstudio-2020-06-3-win64/sifive/riscv64-unknown-elf-gcc-8.3.0-2020.04.1/bin/../lib/gcc/riscv64-unknown-elf/8.3.0/../../../../riscv64-unknown-elf/bin/ld.exe: warning: cannot find entry symbol _start; defaulting to 0000000000000000

$ riscv64-unknown-elf-objcopy -O binary add.elf add.bin
```