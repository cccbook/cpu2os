	.section	__TEXT,__text,regular,pure_instructions
	.macosx_version_min 10, 13
	.globl	_fib
	.p2align	4, 0x90
_fib:                                   ## @fib
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
	subq	$32, %rsp
	movl	%edi, -8(%rbp)
	movl	$1, -12(%rbp)
	movl	-12(%rbp), %edi
	addl	$1, %edi
	movl	%edi, -16(%rbp)
	movl	-8(%rbp), %edi
	addl	$1, %edi
	movl	%edi, -20(%rbp)
	cmpl	$1, -8(%rbp)
	jg	LBB0_2
## BB#1:
	movl	$1, -4(%rbp)
	jmp	LBB0_3
LBB0_2:
	movl	-8(%rbp), %eax
	subl	$1, %eax
	movl	%eax, %edi
	callq	_fib
	movl	-8(%rbp), %edi
	subl	$2, %edi
	movl	%eax, -24(%rbp)         ## 4-byte Spill
	callq	_fib
	movl	-24(%rbp), %edi         ## 4-byte Reload
	addl	%eax, %edi
	movl	%edi, -4(%rbp)
LBB0_3:
	movl	-4(%rbp), %eax
	addq	$32, %rsp
	popq	%rbp
	retq
	.cfi_endproc


.subsections_via_symbols
