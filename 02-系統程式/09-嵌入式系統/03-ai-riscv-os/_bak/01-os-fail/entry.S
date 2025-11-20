.section .text.init
.globl _start

_start:
    # Set up the initial stack pointer
    la sp, __stack_top
    # Jump to the C main function
    call main

# Infinite loop in case main returns
_hang:
    j _hang
