.align 2
.section .text
.globl _start

_start:
        # write(1, message, 13)
        li      a0, 1                # system call 1 is write
        li      a1, 1                # file handle 1 is stdout
        la      a2, message          # address of string to output
        li      a3, 13               # number of bytes
        ecall                        # invoke operating system to do the write

        # exit(0)
        li      a0, 60               # system call 60 is exit
        li      a1, 0                # we want return code 0
        ecall                        # invoke operating system to exit

.section .rodata
message:
        .ascii  "Hello, world\n"
