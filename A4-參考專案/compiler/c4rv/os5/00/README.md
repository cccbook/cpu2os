# 00 -- RISC-V 基礎

* [RISC-V - Getting Started Guide](https://risc-v-getting-started-guide.readthedocs.io/en/latest/)

```
user@DESKTOP-96FRN6B MINGW64 /d/ccc/ccc109/sp/_my/os5/00/riscv-hello-asm (master)
$ ./hello.sh
hello: ELF 64-bit LSB executable, UCB RISC-V, version 1 (SYSV), statically linked, not stripped
HHello.
ello.  
QEMU: Terminated

user@DESKTOP-96FRN6B MINGW64 /d/ccc/ccc109/sp/_my/os5/00/riscv-hello-asm (master)
$ ./hello.sh
hello: ELF 64-bit LSB executable, UCB RISC-V, version 1 (SYSV), statically linked, not stripped
HHello.
ello.  
QEMU: Terminated
```

## riscv-probe

```
user@DESKTOP-96FRN6B MINGW64 /d/ccc/ccc109/sp/_my/os5/00/riscv-probe (master)
$ make
...

user@DESKTOP-96FRN6B MINGW64 /d/ccc/ccc109/sp/_my/os5/00/riscv-probe (master)
$ qemu-system-riscv32 -nographic -machine spike_v1.10 -kernel build/bin/rv32imac/spike/hello
C:\Program Files\qemu\qemu-system-riscv32.exe: info: The Spike v1.10.0 machine has been deprecated. Please use the generic spike machine and specify the ISA versions using -cpu.
hello
```
