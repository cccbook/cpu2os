OUTPUT_ARCH(riscv)
ENTRY(_start)

SECTIONS
{
    . = 0x80000000;

    .text : {
        *(.text.init)
        *(.text .text.*)
    }

    .rodata : {
        *(.rodata .rodata.*)
    }

    .data : {
        *(.data .data.*)
    }

    .bss : {
        . = ALIGN(8);
        __bss_start = .;
        *(.bss .bss.*)
        __bss_end = .;
    }

    . = ALIGN(16);
    .stack : {
        . = . + 4096 * 4;
        __stack_top = .;
    }

    /DISCARD/ : {
        *(.eh_frame)
    }
}