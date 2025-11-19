/* linker.ld */
ENTRY(_start)
OUTPUT_ARCH(riscv)

MEMORY
{
  /* QEMU 'virt' 機器的 RAM */
  ram (rwx) : ORIGIN = 0x80000000, LENGTH = 128M
}

SECTIONS
{
  .text : {
    *(.text.boot) /* 啟動程式碼最優先 */
    *(.text*)
  } > ram

  .rodata : {
    *(.rodata*)
  } > ram

  .data : {
    *(.data*)
  } > ram

  .bss : {
    *(.bss*)
  } > ram

  /* 定義主堆疊 (給 OS 核心, _start 時使用) */
  . = ALIGN(16);
  . += 4K;
  _stack_top = .;
}
