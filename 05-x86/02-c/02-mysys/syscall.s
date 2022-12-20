        .global _write
        .text
_write:
        # write(fd, msg, len)
        mov     $1, %rax                # system call 1 is write
        syscall                         # invoke operating system to do the write
        ret

