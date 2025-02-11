# shecc

## Ubuntu Linux build 

```
root@localhost:/home/guest# cd shecc
root@localhost:/home/guest/shecc# ls
AUTHORS  lib  LICENSE  Makefile  mk  README.md  src  tests  tools
root@localhost:/home/guest/shecc# make
  CC+LD out/inliner
  GEN   out/libc.inc
  CC    out/src/main.o
  LD    out/shecc
  SHECC out/shecc-stage1.elf
  SHECC out/shecc-stage2.elf
qemu: uncaught target signal 11 (Segmentation fault) - core dumped
Makefile:57: recipe for target 'out/shecc-stage2.elf' failed
make: *** [out/shecc-stage2.elf] Segmentation fault (core dumped)
root@localhost:/home/guest/shecc# ls
AUTHORS  lib  LICENSE  Makefile  mk  out  README.md  src  tests  tools        
root@localhost:/home/guest/shecc# ls out
inliner  libc.inc  shecc  shecc-stage1.elf  shecc-stage1.log  src  tests      
root@localhost:/home/guest/shecc# cd out
root@localhost:/home/guest/shecc/out# ./shecc -o fib tests/fib.c
Segmentation fault (core dumped)
root@localhost:/home/guest/shecc/out# cd ..
root@localhost:/home/guest/shecc# out/shecc -o fib tests/fib.c
root@localhost:/home/guest/shecc# chmod +x fib
root@localhost:/home/guest/shecc# ls
AUTHORS  fib  lib  LICENSE  Makefile  mk  out  README.md  src  tests  tools   
root@localhost:/home/guest/shecc# ./fib
F(10) = 55
root@localhost:/home/guest/shecc# qemu-arm fib
F(10) = 55
root@localhost:/home/guest/shecc#
```

## windows build fail

```
user@DESKTOP-96FRN6B MINGW64 /d/ccc/ccc109/sp/_project/shecc (master)
$ make VERBOSE=1
gcc  -O -g -ansi -pedantic -Wall -Wextra -o out/inliner tools/inliner.c
out/inliner lib/c.c out/libc.inc
gcc  -o out/src/main.o -O -g -ansi -pedantic -Wall -Wextra -c -MMD -MF out/src/main.o.d src/main.c
In file included from src/main.c:23:0:
src/../out/libc.inc: In function 'libc_generate':
src/../out/libc.inc:8:7: warning: missing terminating " character
\n");c("#define NULL 0
       ^
src/../out/libc.inc:8:3: error: missing terminating " character
\n");c("#define NULL 0
   ^
src/../out/libc.inc:9:1: error: stray '\' in program
\n");c("
 ^
```