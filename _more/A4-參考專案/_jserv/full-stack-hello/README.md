# full-stack-hello

minimal instruction set and assembler/compiler for "Hello World" execution

* https://github.com/jserv/full-stack-hello

## run (ccc)

```
guest2@localhost:~$ git clone https://github.com/jserv/full-stack-hello
Cloning into 'full-stack-hello'...
remote: Enumerating objects: 409, done.
remote: Counting objects: 100% (3/3), done.
remote: Compressing objects: 100% (3/3), done.
remote: Total 409 (delta 0), reused 1 (delta 0), pack-reused 406
Receiving objects: 100% (409/409), 79.20 KiB | 2.03 MiB/s, done.
Resolving deltas: 100% (221/221), done.
guest2@localhost:~$ cd full-stack-hello/
guest2@localhost:~/full-stack-hello$ ls
as.c  AUTHORS   elf.c  hash.c  LICENSE   opcode.c    private.h  scripts  vm.c          vm.h
as.h  driver.c  elf.h  hash.h  Makefile  opcode.def  README.md  tests    vm_codegen.h
guest2@localhost:~/full-stack-hello$ make

Git commit hooks are installed successfully.

cc -Wall -std=gnu99 -g -fno-crossjumping -c -o vm.o -MMD -MF .vm.o.d vm.c
cc -Wall -std=gnu99 -g -fno-crossjumping -c -o as.o -MMD -MF .as.o.d as.c
cc -Wall -std=gnu99 -g -fno-crossjumping -c -o opcode.o -MMD -MF .opcode.o.d opcode.c
cc -Wall -std=gnu99 -g -fno-crossjumping -c -o driver.o -MMD -MF .driver.o.d driver.c
cc -Wall -std=gnu99 -g -fno-crossjumping -c -o elf.o -MMD -MF .elf.o.d elf.c
cc -Wall -std=gnu99 -g -fno-crossjumping -c -o hash.o -MMD -MF .hash.o.d hash.c
hash.c: In function ‘hash_djb2’:
hash.c:6:12: warning: suggest parentheses around assignment used as truth value [-Wparentheses]
     while (c = *str) {
            ^
cc -Wall -std=gnu99 -g -fno-crossjumping -o as_exec vm.o as.o opcode.o driver.o elf.o hash.o
guest2@localhost:~/full-stack-hello$ make check
5
1
3
6
2
4
7
8
*** tests/label.s *** [ Verified ]
42
*** tests/halt.s *** [ Verified ]
Hello World
*** tests/hello.s *** [ Verified ]
42
50
150
-4
80
2
1
66
77
1
2
1
-1
-2
2147483647
-1073741824
*** tests/coverage.s *** [ Verified ]
cond_jump_insts_ok
*** tests/jcond.s *** [ Verified ]
6
-6
-6
4
6
-18
18
36
-24
-864
*** tests/mul.s *** [ Verified ]
guest2@localhost:~/full-stack-hello$ ls
as.c     as.o      driver.o  elf.o   hash.o    opcode.c    opcode.o   scripts  vm_codegen.h
as_exec  AUTHORS   elf.c     hash.c  LICENSE   opcode.def  private.h  tests    vm.h
as.h     driver.c  elf.h     hash.h  Makefile  opcode.h    README.md  vm.c     vm.o
guest2@localhost:~/full-stack-hello$ ./as_exec -h
Usage: as_exec [-w] [-x] [-o <out_file>] <in_file>
       -w Assemble <in_file> and write to an ELF file, see -o below
       -o if -w is specifed, <out_file> is used to store the object code
       -x Load <in_file> and execute it

       <in_file> the file name to be used by commands above
guest2@localhost:~/full-stack-hello$ ./as_exec tests/hello.s
Hello World
guest2@localhost:~/full-stack-hello$ ./as_exec -w tests/hello.s
guest2@localhost:~/full-stack-hello$ ./as_exec -o tests/temp.o -w tests/hello.s
guest2@localhost:~/full-stack-hello$ ./as_exec -x tests/hello.o
Segmentation fault (core dumped)
guest2@localhost:~/full-stack-hello$ objdump -x tests/hello.o

tests/hello.o:     file format elf64-little
tests/hello.o
architecture: UNKNOWN!, flags 0x00000102:
EXEC_P, D_PAGED
start address 0x0000000000000000

Program Header:
0x464c457f off    0x0000000000000000 vaddr 0x0000000100000002 paddr 0x0000000000000000 align 2**54
         filesz 0x0000000000000000 memsz 0x0000000000000000 flags -w- 10100
  INTERP off    0x0000000000000001 vaddr 0x00000000000000e8 paddr 0x0000000000000000 align 2**5
         filesz 0x0000000000000000 memsz 0x0000000000000018 flags ---
    0x20 off    0x0000000000000001 vaddr 0x0000000000000100 paddr 0x0000000000000000 align 2**5
         filesz 0x0000000000000000 memsz 0x000000000000001c flags ---

Sections:
Idx Name          Size      VMA               LMA               File off  Algn
SYMBOL TABLE:
no symbols
```