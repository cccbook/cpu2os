# riscv-probe

來源 -- https://github.com/michaeljclark/riscv-probe

## make & run

在 windows 上只有 virt 可以測試成功！

```
$ make
...

user@DESKTOP-96FRN6B MINGW64 /d/ccc/ccc109/sp/10-riscv/07-riscv-probe (master)
$ qemu-system-riscv32 -nographic -machine virt -bios none -kernel build/bin/rv32imac/virt/hello       
hello

user@DESKTOP-96FRN6B MINGW64 /d/ccc/ccc109/sp/10-riscv/07-riscv-probe (master)
$ qemu-system-riscv64 -nographic -machine virt -bios none -kernel build/bin/rv64imac/virt/hello       
hello
```

## 32bit machine:virt / all example

```
user@DESKTOP-96FRN6B MINGW64 /d/ccc/ccc109/sp/10-riscv/07-riscv-probe (master)
$ qemu-system-riscv32 -bios none -nographic -machine virt -kernel build/bin/rv32imac/virt/hello       
hello

user@DESKTOP-96FRN6B MINGW64 /d/ccc/ccc109/sp/10-riscv/07-riscv-probe (master)
$ qemu-system-riscv32 -bios none -nographic -machine virt -kernel build/bin/rv32imac/virt/abort
abort

user@DESKTOP-96FRN6B MINGW64 /d/ccc/ccc109/sp/10-riscv/07-riscv-probe (master)
$ qemu-system-riscv32 -bios none -nographic -machine virt -kernel build/bin/rv32imac/virt/alloca
alloca[0]=0x800013a0
alloca[1]=0x80001310
alloca[2]=0x80001280
alloca[3]=0x800011f0

user@DESKTOP-96FRN6B MINGW64 /d/ccc/ccc109/sp/10-riscv/07-riscv-probe (master)
$ qemu-system-riscv32 -bios none -nographic -machine virt -kernel build/bin/rv32imac/virt/argv0
libfemto/arch/riscv/trap.c:63: machine mode: unhandlable trap 5 @ 0x8000028e

user@DESKTOP-96FRN6B MINGW64 /d/ccc/ccc109/sp/10-riscv/07-riscv-probe (master)
$ qemu-system-riscv32 -bios none -nographic -machine virt -kernel build/bin/rv32imac/virt/enclave
text: 0x80000000 - 0x80000fff
data: 0x80001000 - 0x80002fff
uart: 0x10000000 - 0x1000001f
pmp.count: 16
pmp.width: 34
pmp.granularity: 2
riscv-enclave
libfemto/arch/riscv/trap.c:63: machine mode: unhandlable trap 7 @ 0x800005b0

user@DESKTOP-96FRN6B MINGW64 /d/ccc/ccc109/sp/10-riscv/07-riscv-probe (master)
$ qemu-system-riscv32 -bios none -nographic -machine virt -kernel build/bin/rv32imac/virt/hang
hang
QEMU: Terminated

user@DESKTOP-96FRN6B MINGW64 /d/ccc/ccc109/sp/10-riscv/07-riscv-probe (master)
$ qemu-system-riscv32 -bios none -nographic -machine virt -kernel build/bin/rv32imac/virt/malloc
malloc[0]=0x8000207c
malloc[1]=0x80002108
malloc[2]=0x80002194
malloc[3]=0x80002220
malloc[4]=0x800022ac
malloc[5]=0x80002338
malloc[6]=0x800023c4
malloc[7]=0x80002450

user@DESKTOP-96FRN6B MINGW64 /d/ccc/ccc109/sp/10-riscv/07-riscv-probe (master)
$ qemu-system-riscv32 -bios none -nographic -machine virt -kernel build/bin/rv32imac/virt/memory
memory_start=0x80000000 - 0x87ffffff

user@DESKTOP-96FRN6B MINGW64 /d/ccc/ccc109/sp/10-riscv/07-riscv-probe (master)
$ qemu-system-riscv32 -bios none -nographic -machine virt -kernel build/bin/rv32imac/virt/probe
isa: rv32imafdcsu
csr: fflags          illegal_instruction cause=2 mtval=0x00000000
csr: frm             illegal_instruction cause=2 mtval=0x00000000
csr: fcsr            illegal_instruction cause=2 mtval=0x00000000
csr: mcycle          0xdeccc37e
csr: minstret        0xf1256dfa
csr: mcycleh         0x000049af
csr: minstreth       0x000049b0
csr: cycle           0x1f236bf8
csr: time            illegal_instruction cause=2 mtval=0x00000000
csr: instret         0x3f6b94e4
csr: cycleh          0x000049b0
csr: timeh           illegal_instruction cause=2 mtval=0x00000000
csr: instreth        0x000049b0
csr: mvendorid       0x00000000
csr: marchid         0x00000000
csr: mimpid          0x00000000
csr: mhartid         0x00000000
csr: mstatus         0x00000000
csr: misa            0x4014112d
csr: medeleg         0x00000000
csr: mideleg         0x00000000
csr: mie             0x00000000
csr: mtvec           0x8000002c
csr: mcounteren      0x00000000
csr: mscratch        0x00000000
csr: mepc            0x8000026c
csr: mcause          0x00000002
csr: mtval           0x00000000
csr: mip             0x00000000
csr: sstatus         0x00000000
csr: sedeleg         illegal_instruction cause=2 mtval=0x00000000
csr: sideleg         illegal_instruction cause=2 mtval=0x00000000
csr: sie             0x00000000
csr: stvec           0x00000000
csr: scounteren      0x00000000
csr: sscratch        0x00000000
csr: sepc            0x00000000
csr: scause          0x00000000
csr: stval           0x00000000
csr: sip             0x00000000
csr: satp            0x00000000
csr: pmpcfg0         0x00000000
csr: pmpcfg1         0x00000000
csr: pmpcfg2         0x00000000
csr: pmpcfg3         0x00000000
csr: pmpaddr0        0x00000000
csr: pmpaddr1        0x00000000
csr: pmpaddr2        0x00000000
csr: pmpaddr3        0x00000000
csr: pmpaddr4        0x00000000
csr: pmpaddr5        0x00000000
csr: pmpaddr6        0x00000000
csr: pmpaddr7        0x00000000
csr: pmpaddr8        0x00000000
csr: pmpaddr9        0x00000000
csr: pmpaddr10       0x00000000
csr: pmpaddr11       0x00000000
csr: pmpaddr12       0x00000000
csr: pmpaddr13       0x00000000
csr: pmpaddr14       0x00000000
csr: pmpaddr15       0x00000000


user@DESKTOP-96FRN6B MINGW64 /d/ccc/ccc109/sp/10-riscv/07-riscv-probe (master)
$ qemu-system-riscv32 -bios none -nographic -machine virt -kernel build/bin/rv32imac/virt/symbols
_text_start=0x80000000
_text_end=0x8000057a
_rodata_start=0x8000057c
_rodata_end=0x8000068c
_data_start=0x80001000
_data_end=0x80001060
_bss_start=0x80001060
_bss_end=0x80002070
_memory_start=0x80000000
_memory_end=0x88000000

user@DESKTOP-96FRN6B MINGW64 /d/ccc/ccc109/sp/10-riscv/07-riscv-probe (master)
$ qemu-system-riscv32 -bios none -nographic -machine virt -kernel build/bin/rv32imac/virt/user
riscv-user-mode
```

## make log

```
user@DESKTOP-96FRN6B MINGW64 /d/ccc/ccc109/sp/_my/os5/00/riscv-probe (master)
$ make
AS.rv32imac build/obj/rv32imac/env/default/crt.o
CC.rv32imac build/obj/rv32imac/env/default/setup.o
CC.rv32imac build/obj/rv32imac/examples/abort/main.o
CC.rv32imac build/obj/rv32imac/libfemto/arch/riscv/auxval.o
CC.rv32imac build/obj/rv32imac/libfemto/arch/riscv/csr.o
CC.rv32imac build/obj/rv32imac/libfemto/arch/riscv/device.o
CC.rv32imac build/obj/rv32imac/libfemto/arch/riscv/memory.o
CC.rv32imac build/obj/rv32imac/libfemto/arch/riscv/pmp.o
CC.rv32imac build/obj/rv32imac/libfemto/arch/riscv/spinlock.o
CC.rv32imac build/obj/rv32imac/libfemto/arch/riscv/start.o
CC.rv32imac build/obj/rv32imac/libfemto/arch/riscv/trap.o
CC.rv32imac build/obj/rv32imac/libfemto/drivers/htif.o
CC.rv32imac build/obj/rv32imac/libfemto/drivers/ns16550a.o
CC.rv32imac build/obj/rv32imac/libfemto/drivers/semihost.o
CC.rv32imac build/obj/rv32imac/libfemto/drivers/sifive_test.o
CC.rv32imac build/obj/rv32imac/libfemto/drivers/sifive_uart.o
CC.rv32imac build/obj/rv32imac/libfemto/std/abort.o
CC.rv32imac build/obj/rv32imac/libfemto/std/clz.o
CC.rv32imac build/obj/rv32imac/libfemto/std/ctz.o
CC.rv32imac build/obj/rv32imac/libfemto/std/exit.o
CC.rv32imac build/obj/rv32imac/libfemto/std/getchar.o
CC.rv32imac build/obj/rv32imac/libfemto/std/malloc.o
CC.rv32imac build/obj/rv32imac/libfemto/std/memchr.o
CC.rv32imac build/obj/rv32imac/libfemto/std/memcmp.o
CC.rv32imac build/obj/rv32imac/libfemto/std/memcpy.o
CC.rv32imac build/obj/rv32imac/libfemto/std/memset.o
CC.rv32imac build/obj/rv32imac/libfemto/std/printf.o
CC.rv32imac build/obj/rv32imac/libfemto/std/putchar.o
CC.rv32imac build/obj/rv32imac/libfemto/std/puts.o
CC.rv32imac build/obj/rv32imac/libfemto/std/snprintf.o
CC.rv32imac build/obj/rv32imac/libfemto/std/strchr.o
CC.rv32imac build/obj/rv32imac/libfemto/std/strcmp.o
CC.rv32imac build/obj/rv32imac/libfemto/std/strlen.o
CC.rv32imac build/obj/rv32imac/libfemto/std/strncmp.o
CC.rv32imac build/obj/rv32imac/libfemto/std/strncpy.o
CC.rv32imac build/obj/rv32imac/libfemto/std/vprintf.o
CC.rv32imac build/obj/rv32imac/libfemto/std/vsnprintf.o
AR.rv32imac build/lib/rv32imac/libfemto.a
LD.rv32imac build/bin/rv32imac/default/abort
AS.rv64imac build/obj/rv64imac/env/default/crt.o
CC.rv64imac build/obj/rv64imac/env/default/setup.o
CC.rv64imac build/obj/rv64imac/examples/abort/main.o
CC.rv64imac build/obj/rv64imac/libfemto/arch/riscv/auxval.o
CC.rv64imac build/obj/rv64imac/libfemto/arch/riscv/csr.o
CC.rv64imac build/obj/rv64imac/libfemto/arch/riscv/device.o
CC.rv64imac build/obj/rv64imac/libfemto/arch/riscv/memory.o
CC.rv64imac build/obj/rv64imac/libfemto/arch/riscv/pmp.o
CC.rv64imac build/obj/rv64imac/libfemto/arch/riscv/spinlock.o
CC.rv64imac build/obj/rv64imac/libfemto/arch/riscv/start.o
CC.rv64imac build/obj/rv64imac/libfemto/arch/riscv/trap.o
CC.rv64imac build/obj/rv64imac/libfemto/drivers/htif.o
CC.rv64imac build/obj/rv64imac/libfemto/drivers/ns16550a.o
CC.rv64imac build/obj/rv64imac/libfemto/drivers/semihost.o
CC.rv64imac build/obj/rv64imac/libfemto/drivers/sifive_test.o
CC.rv64imac build/obj/rv64imac/libfemto/drivers/sifive_uart.o
CC.rv64imac build/obj/rv64imac/libfemto/std/abort.o
CC.rv64imac build/obj/rv64imac/libfemto/std/clz.o
CC.rv64imac build/obj/rv64imac/libfemto/std/ctz.o
CC.rv64imac build/obj/rv64imac/libfemto/std/exit.o
CC.rv64imac build/obj/rv64imac/libfemto/std/getchar.o
CC.rv64imac build/obj/rv64imac/libfemto/std/malloc.o
CC.rv64imac build/obj/rv64imac/libfemto/std/memchr.o
CC.rv64imac build/obj/rv64imac/libfemto/std/memcmp.o
CC.rv64imac build/obj/rv64imac/libfemto/std/memcpy.o
CC.rv64imac build/obj/rv64imac/libfemto/std/memset.o
CC.rv64imac build/obj/rv64imac/libfemto/std/printf.o
CC.rv64imac build/obj/rv64imac/libfemto/std/putchar.o
CC.rv64imac build/obj/rv64imac/libfemto/std/puts.o
CC.rv64imac build/obj/rv64imac/libfemto/std/snprintf.o
CC.rv64imac build/obj/rv64imac/libfemto/std/strchr.o
CC.rv64imac build/obj/rv64imac/libfemto/std/strcmp.o
CC.rv64imac build/obj/rv64imac/libfemto/std/strlen.o
CC.rv64imac build/obj/rv64imac/libfemto/std/strncmp.o
CC.rv64imac build/obj/rv64imac/libfemto/std/strncpy.o
CC.rv64imac build/obj/rv64imac/libfemto/std/vprintf.o
CC.rv64imac build/obj/rv64imac/libfemto/std/vsnprintf.o
AR.rv64imac build/lib/rv64imac/libfemto.a
LD.rv64imac build/bin/rv64imac/default/abort
AS.rv32imac build/obj/rv32imac/env/spike/crt.o
CC.rv32imac build/obj/rv32imac/env/spike/setup.o
LD.rv32imac build/bin/rv32imac/spike/abort
AS.rv64imac build/obj/rv64imac/env/spike/crt.o
CC.rv64imac build/obj/rv64imac/env/spike/setup.o
LD.rv64imac build/bin/rv64imac/spike/abort
AS.rv32imac build/obj/rv32imac/env/virt/crt.o
CC.rv32imac build/obj/rv32imac/env/virt/setup.o
LD.rv32imac build/bin/rv32imac/virt/abort
AS.rv64imac build/obj/rv64imac/env/virt/crt.o
CC.rv64imac build/obj/rv64imac/env/virt/setup.o
LD.rv64imac build/bin/rv64imac/virt/abort
AS.rv32imac build/obj/rv32imac/env/qemu-sifive_e/crt.o
CC.rv32imac build/obj/rv32imac/env/qemu-sifive_e/setup.o
LD.rv32imac build/bin/rv32imac/qemu-sifive_e/abort
AS.rv64imac build/obj/rv64imac/env/qemu-sifive_e/crt.o
CC.rv64imac build/obj/rv64imac/env/qemu-sifive_e/setup.o
LD.rv64imac build/bin/rv64imac/qemu-sifive_e/abort
AS.rv32imac build/obj/rv32imac/env/qemu-sifive_u/crt.o
CC.rv32imac build/obj/rv32imac/env/qemu-sifive_u/setup.o
LD.rv32imac build/bin/rv32imac/qemu-sifive_u/abort
AS.rv64imac build/obj/rv64imac/env/qemu-sifive_u/crt.o
CC.rv64imac build/obj/rv64imac/env/qemu-sifive_u/setup.o
LD.rv64imac build/bin/rv64imac/qemu-sifive_u/abort
AS.rv32imac build/obj/rv32imac/env/coreip-e2-arty/crt.o
CC.rv32imac build/obj/rv32imac/env/coreip-e2-arty/setup.o
LD.rv32imac build/bin/rv32imac/coreip-e2-arty/abort
CC.rv32imac build/obj/rv32imac/examples/alloca/main.o
LD.rv32imac build/bin/rv32imac/default/alloca
CC.rv64imac build/obj/rv64imac/examples/alloca/main.o
LD.rv64imac build/bin/rv64imac/default/alloca
LD.rv32imac build/bin/rv32imac/spike/alloca
LD.rv64imac build/bin/rv64imac/spike/alloca
LD.rv32imac build/bin/rv32imac/virt/alloca
LD.rv64imac build/bin/rv64imac/virt/alloca
LD.rv32imac build/bin/rv32imac/qemu-sifive_e/alloca
LD.rv64imac build/bin/rv64imac/qemu-sifive_e/alloca
LD.rv32imac build/bin/rv32imac/qemu-sifive_u/alloca
LD.rv64imac build/bin/rv64imac/qemu-sifive_u/alloca
LD.rv32imac build/bin/rv32imac/coreip-e2-arty/alloca
CC.rv32imac build/obj/rv32imac/examples/argv0/main.o
LD.rv32imac build/bin/rv32imac/default/argv0
CC.rv64imac build/obj/rv64imac/examples/argv0/main.o
LD.rv64imac build/bin/rv64imac/default/argv0
LD.rv32imac build/bin/rv32imac/spike/argv0
LD.rv64imac build/bin/rv64imac/spike/argv0
LD.rv32imac build/bin/rv32imac/virt/argv0
LD.rv64imac build/bin/rv64imac/virt/argv0
LD.rv32imac build/bin/rv32imac/qemu-sifive_e/argv0
LD.rv64imac build/bin/rv64imac/qemu-sifive_e/argv0
LD.rv32imac build/bin/rv32imac/qemu-sifive_u/argv0
LD.rv64imac build/bin/rv64imac/qemu-sifive_u/argv0
LD.rv32imac build/bin/rv32imac/coreip-e2-arty/argv0
CC.rv32imac build/obj/rv32imac/examples/enclave/enclave.o
LD.rv32imac build/bin/rv32imac/default/enclave
CC.rv64imac build/obj/rv64imac/examples/enclave/enclave.o
LD.rv64imac build/bin/rv64imac/default/enclave
LD.rv32imac build/bin/rv32imac/spike/enclave
LD.rv64imac build/bin/rv64imac/spike/enclave
LD.rv32imac build/bin/rv32imac/virt/enclave
LD.rv64imac build/bin/rv64imac/virt/enclave
LD.rv32imac build/bin/rv32imac/qemu-sifive_e/enclave
LD.rv64imac build/bin/rv64imac/qemu-sifive_e/enclave
LD.rv32imac build/bin/rv32imac/qemu-sifive_u/enclave
LD.rv64imac build/bin/rv64imac/qemu-sifive_u/enclave
LD.rv32imac build/bin/rv32imac/coreip-e2-arty/enclave
CC.rv32imac build/obj/rv32imac/examples/hang/hang.o
LD.rv32imac build/bin/rv32imac/default/hang
CC.rv64imac build/obj/rv64imac/examples/hang/hang.o
LD.rv64imac build/bin/rv64imac/default/hang
LD.rv32imac build/bin/rv32imac/spike/hang
LD.rv64imac build/bin/rv64imac/spike/hang
LD.rv32imac build/bin/rv32imac/virt/hang
LD.rv64imac build/bin/rv64imac/virt/hang
LD.rv32imac build/bin/rv32imac/qemu-sifive_e/hang
LD.rv64imac build/bin/rv64imac/qemu-sifive_e/hang
LD.rv32imac build/bin/rv32imac/qemu-sifive_u/hang
LD.rv64imac build/bin/rv64imac/qemu-sifive_u/hang
LD.rv32imac build/bin/rv32imac/coreip-e2-arty/hang
CC.rv32imac build/obj/rv32imac/examples/hello/hello.o
LD.rv32imac build/bin/rv32imac/default/hello
CC.rv64imac build/obj/rv64imac/examples/hello/hello.o
LD.rv64imac build/bin/rv64imac/default/hello
LD.rv32imac build/bin/rv32imac/spike/hello
LD.rv64imac build/bin/rv64imac/spike/hello
LD.rv32imac build/bin/rv32imac/virt/hello
LD.rv64imac build/bin/rv64imac/virt/hello
LD.rv32imac build/bin/rv32imac/qemu-sifive_e/hello
LD.rv64imac build/bin/rv64imac/qemu-sifive_e/hello
LD.rv32imac build/bin/rv32imac/qemu-sifive_u/hello
LD.rv64imac build/bin/rv64imac/qemu-sifive_u/hello
LD.rv32imac build/bin/rv32imac/coreip-e2-arty/hello
CC.rv32imac build/obj/rv32imac/examples/malloc/main.o
LD.rv32imac build/bin/rv32imac/default/malloc
CC.rv64imac build/obj/rv64imac/examples/malloc/main.o
LD.rv64imac build/bin/rv64imac/default/malloc
LD.rv32imac build/bin/rv32imac/spike/malloc
LD.rv64imac build/bin/rv64imac/spike/malloc
LD.rv32imac build/bin/rv32imac/virt/malloc
LD.rv64imac build/bin/rv64imac/virt/malloc
LD.rv32imac build/bin/rv32imac/qemu-sifive_e/malloc
LD.rv64imac build/bin/rv64imac/qemu-sifive_e/malloc
LD.rv32imac build/bin/rv32imac/qemu-sifive_u/malloc
LD.rv64imac build/bin/rv64imac/qemu-sifive_u/malloc
LD.rv32imac build/bin/rv32imac/coreip-e2-arty/malloc
CC.rv32imac build/obj/rv32imac/examples/memory/main.o
LD.rv32imac build/bin/rv32imac/default/memory
CC.rv64imac build/obj/rv64imac/examples/memory/main.o
LD.rv64imac build/bin/rv64imac/default/memory
LD.rv32imac build/bin/rv32imac/spike/memory
LD.rv64imac build/bin/rv64imac/spike/memory
LD.rv32imac build/bin/rv32imac/virt/memory
LD.rv64imac build/bin/rv64imac/virt/memory
LD.rv32imac build/bin/rv32imac/qemu-sifive_e/memory
LD.rv64imac build/bin/rv64imac/qemu-sifive_e/memory
LD.rv32imac build/bin/rv32imac/qemu-sifive_u/memory
LD.rv64imac build/bin/rv64imac/qemu-sifive_u/memory
LD.rv32imac build/bin/rv32imac/coreip-e2-arty/memory
CC.rv32imac build/obj/rv32imac/examples/probe/probe.o
LD.rv32imac build/bin/rv32imac/default/probe
CC.rv64imac build/obj/rv64imac/examples/probe/probe.o
LD.rv64imac build/bin/rv64imac/default/probe
LD.rv32imac build/bin/rv32imac/spike/probe
LD.rv64imac build/bin/rv64imac/spike/probe
LD.rv32imac build/bin/rv32imac/virt/probe
LD.rv64imac build/bin/rv64imac/virt/probe
LD.rv32imac build/bin/rv32imac/qemu-sifive_e/probe
LD.rv64imac build/bin/rv64imac/qemu-sifive_e/probe
LD.rv32imac build/bin/rv32imac/qemu-sifive_u/probe
LD.rv64imac build/bin/rv64imac/qemu-sifive_u/probe
LD.rv32imac build/bin/rv32imac/coreip-e2-arty/probe
CC.rv32imac build/obj/rv32imac/examples/symbols/symbols.o
LD.rv32imac build/bin/rv32imac/default/symbols
CC.rv64imac build/obj/rv64imac/examples/symbols/symbols.o
LD.rv64imac build/bin/rv64imac/default/symbols
LD.rv32imac build/bin/rv32imac/spike/symbols
LD.rv64imac build/bin/rv64imac/spike/symbols
LD.rv32imac build/bin/rv32imac/virt/symbols
LD.rv64imac build/bin/rv64imac/virt/symbols
LD.rv32imac build/bin/rv32imac/qemu-sifive_e/symbols
LD.rv64imac build/bin/rv64imac/qemu-sifive_e/symbols
LD.rv32imac build/bin/rv32imac/qemu-sifive_u/symbols
LD.rv64imac build/bin/rv64imac/qemu-sifive_u/symbols
LD.rv32imac build/bin/rv32imac/coreip-e2-arty/symbols
CC.rv32imac build/obj/rv32imac/examples/user/user.o
LD.rv32imac build/bin/rv32imac/default/user
CC.rv64imac build/obj/rv64imac/examples/user/user.o
LD.rv64imac build/bin/rv64imac/default/user
LD.rv32imac build/bin/rv32imac/spike/user
LD.rv64imac build/bin/rv64imac/spike/user
LD.rv32imac build/bin/rv32imac/virt/user
LD.rv64imac build/bin/rv64imac/virt/user
LD.rv32imac build/bin/rv32imac/qemu-sifive_e/user
LD.rv64imac build/bin/rv64imac/qemu-sifive_e/user
LD.rv32imac build/bin/rv32imac/qemu-sifive_u/user
LD.rv64imac build/bin/rv64imac/qemu-sifive_u/user
LD.rv32imac build/bin/rv32imac/coreip-e2-arty/user
```