


```
guest2@localhost:~$ cd riscv2os
guest2@localhost:~/riscv2os$ ls
doc  fs  LICENSE  linux  mini-riscv-os  mylib  README.md  semu  test  vm  xv6  xv7
guest2@localhost:~/riscv2os$ cd mini-riscv-os/
guest2@localhost:~/riscv2os/mini-riscv-os$ ls
01-HelloOs        03-MultiTasking    05b-Preemptive  06-IoTrap      A1-Input    A3-keyInterrupt
02-ContextSwitch  04-TimerInterrupt  05-Preemptive   10-SystemCall  A2-pooling  A6-Net
guest2@localhost:~/riscv2os/mini-riscv-os$ cd 01-HelloOs/
guest2@localhost:~/riscv2os/mini-riscv-os/01-HelloOs$ ls
Makefile  os.c  os.elf  os.ld  README.md  start.s
guest2@localhost:~/riscv2os/mini-riscv-os/01-HelloOs$ make
make: Nothing to be done for 'all'.
guest2@localhost:~/riscv2os/mini-riscv-os/01-HelloOs$ make clean
rm -f *.elf
guest2@localhost:~/riscv2os/mini-riscv-os/01-HelloOs$ make
riscv64-unknown-elf-gcc -nostdlib -fno-builtin -mcmodel=medany -march=rv32ima -mabi=ilp32 -T os.ld -o os.elf start.s os.c
guest2@localhost:~/riscv2os/mini-riscv-os/01-HelloOs$ ls
Makefile  os.c  os.elf  os.ld  README.md  start.s
guest2@localhost:~/riscv2os/mini-riscv-os/01-HelloOs$ make qemu
Press Ctrl-A and then X to exit QEMU
qemu-system-riscv32 -nographic -smp 4 -machine virt -bios none -kernel os.elf
Hello OS!
QEMU: Terminated
guest2@localhost:~/riscv2os/mini-riscv-os/01-HelloOs$ cd ..
guest2@localhost:~/riscv2os/mini-riscv-os$ cd 02-ContextSwitch/
guest2@localhost:~/riscv2os/mini-riscv-os/02-ContextSwitch$ ls
lib.c  lib.h  Makefile  os.c  os.h  os.ld  README.md  riscv.h  start.s  sys.h  sys.s
guest2@localhost:~/riscv2os/mini-riscv-os/02-ContextSwitch$ make clean
rm -f *.elf
guest2@localhost:~/riscv2os/mini-riscv-os/02-ContextSwitch$ make
riscv64-unknown-elf-gcc -nostdlib -fno-builtin -mcmodel=medany -march=rv32ima -mabi=ilp32 -T os.ld -o os.elf start.s sys.s lib.c os.c
guest2@localhost:~/riscv2os/mini-riscv-os/02-ContextSwitch$ ls
lib.c  lib.h  Makefile  os.c  os.elf  os.h  os.ld  README.md  riscv.h  start.s  sys.h  sys.s
guest2@localhost:~/riscv2os/mini-riscv-os/02-ContextSwitch$ make qemu
Press Ctrl-A and then X to exit QEMU
qemu-system-riscv32 -nographic -smp 4 -machine virt -bios none -kernel os.elf
OS start
Task0: Context Switch Success !
QEMU: Terminated
guest2@localhost:~/riscv2os/mini-riscv-os/02-ContextSwitch$ cd ..
guest2@localhost:~/riscv2os/mini-riscv-os$ cd 03-MultiTasking/
guest2@localhost:~/riscv2os/mini-riscv-os/03-MultiTasking$ make clean
rm -f *.elf
guest2@localhost:~/riscv2os/mini-riscv-os/03-MultiTasking$ make
riscv64-unknown-elf-gcc -nostdlib -fno-builtin -mcmodel=medany -march=rv32ima -mabi=ilp32 -T os.ld -o os.elf start.s sys.s lib.c task.c os.c user.c
guest2@localhost:~/riscv2os/mini-riscv-os/03-MultiTasking$ make qemu
Press Ctrl-A and then X to exit QEMU
qemu-system-riscv32 -nographic -smp 4 -machine virt -bios none -kernel os.elf
OS start
OS: Activate next task
Task0: Created!
Task0: Now, return to kernel mode
OS: Back to OS

OS: Activate next task
Task1: Created!
Task1: Now, return to kernel mode
OS: Back to OS

OS: Activate next task
Task0: Running...
OS: Back to OS

OS: Activate next task
Task1: Running...
OS: Back to OS

OS: Activate next task
Task0: Running...
OS: Back to OS

OS: Activate next task
Task1: Running...
OS: Back to OS

OS: Activate next task
Task0: Running...
QEMU: Terminated
guest2@localhost:~/riscv2os/mini-riscv-os/03-MultiTasking$ cd ..
guest2@localhost:~/riscv2os/mini-riscv-os$ cd 04-TimerInterrupt/
guest2@localhost:~/riscv2os/mini-riscv-os/04-TimerInterrupt$ ls
lib.c  Makefile  os.elf  os.ld      riscv.h  sys.h  timer.c
lib.h  os.c      os.h    README.md  start.s  sys.s  timer.h
guest2@localhost:~/riscv2os/mini-riscv-os/04-TimerInterrupt$ make clean
rm -f *.elf
guest2@localhost:~/riscv2os/mini-riscv-os/04-TimerInterrupt$ make
riscv64-unknown-elf-gcc -nostdlib -fno-builtin -mcmodel=medany -march=rv32ima -mabi=ilp32 -T os.ld -o os.elf start.s sys.s lib.c timer.c os.c
guest2@localhost:~/riscv2os/mini-riscv-os/04-TimerInterrupt$ make qemu
Press Ctrl-A and then X to exit QEMU
qemu-system-riscv32 -nographic -smp 4 -machine virt -bios none -kernel os.elf
OS start
timer_handler: 1
timer_handler: 2
timer_handler: 3
QEMU: Terminated
guest2@localhost:~/riscv2os/mini-riscv-os/04-TimerInterrupt$ cd ..
guest2@localhost:~/riscv2os/mini-riscv-os$ ls
01-HelloOs        03-MultiTasking    05b-Preemptive  06-IoTrap      A1-Input    A3-keyInterrupt
02-ContextSwitch  04-TimerInterrupt  05-Preemptive   10-SystemCall  A2-pooling  A6-Net
guest2@localhost:~/riscv2os/mini-riscv-os$ cd 05-Preemptive/
guest2@localhost:~/riscv2os/mini-riscv-os/05-Preemptive$ ls
lib.c  Makefile  os.elf  os.ld      riscv.h  sys.h  task.c  timer.c  user.c
lib.h  os.c      os.h    README.md  start.s  sys.s  task.h  timer.h
guest2@localhost:~/riscv2os/mini-riscv-os/05-Preemptive$ make clean
rm -f *.elf
guest2@localhost:~/riscv2os/mini-riscv-os/05-Preemptive$ make
riscv64-unknown-elf-gcc -nostdlib -fno-builtin -mcmodel=medany -march=rv32ima -mabi=ilp32 -T os.ld -o os.elf start.s sys.s lib.c timer.c task.c os.c user.c
guest2@localhost:~/riscv2os/mini-riscv-os/05-Preemptive$ make qemu
Press Ctrl-A and then X to exit QEMU
qemu-system-riscv32 -nographic -smp 4 -machine virt -bios none -kernel os.elf
OS start
OS: Activate next task
Task0: Created!
Task0: Running...
Task0: Running...
Task0: Running...
Task0: Running...
Task0: Running...
timer_handler: 1
OS: Back to OS

OS: Activate next task
Task1: Created!
Task1: Running...
Task1: Running...
Task1: Running...
Task1: Running...
QEMU: Terminated
```
