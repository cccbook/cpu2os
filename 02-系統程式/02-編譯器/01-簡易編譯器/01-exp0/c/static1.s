	.section	__TEXT,__text,regular,pure_instructions
	.macosx_version_min 10, 13
	.globl	_nextTemp
	.p2align	4, 0x90
_nextTemp:                              ## @nextTemp
	.cfi_startproc
## BB#0:
	pushq	%rbp
Lcfi0:
	.cfi_def_cfa_offset 16
Lcfi1:
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
Lcfi2:
	.cfi_def_cfa_register %rbp
	movl	_nextTemp.tempIdx(%rip), %eax
	movl	%eax, %ecx
	addl	$1, %ecx
	movl	%ecx, _nextTemp.tempIdx(%rip)
	popq	%rbp
	retq
	.cfi_endproc

.zerofill __DATA,__bss,_nextTemp.tempIdx,4,2 ## @nextTemp.tempIdx

.subsections_via_symbols
