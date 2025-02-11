	.file	"hello_bc.c"
	.section	.text.unlikely,"x"
LCOLDB0:
	.text
LHOTB0:
	.p2align 4,,15
	.globl	_run
	.def	_run;	.scl	2;	.type	32;	.endef
_run:
	pushl	%esi
	pushl	%ebx
	movl	$_stack.2652+40000, %esi
	movl	$L2, %ebx
	movl	%esi, %ecx
	movl	$_code.2649, %edx
	subl	$36, %esp
	jmp	*%ebx
	.p2align 4,,10
L10:
L11:
	movl	%esi, %edx
	movl	(%esi), %esi
	leal	8(%edx), %ecx
	movl	4(%edx), %edx
	testl	%edx, %edx
	je	L13
L16:
	movl	%edx, %ebx
	subl	$_code.2649, %ebx
	sarl	$2, %ebx
	movl	__pc.2653(,%ebx,4), %ebx
	addl	$L2, %ebx
	jmp	*%ebx
	.p2align 4,,10
L2:
	addl	$4, %edx
L4:
	movl	%edx, %eax
	movl	%esi, -4(%ecx)
	leal	-4(%ecx), %ebx
	movl	(%eax), %eax
	addl	$4, %edx
	movl	%ebx, %ecx
	movl	%ebx, %esi
	sall	$2, %eax
	subl	%eax, %ecx
L5:
	addl	$4, %edx
L6:
	movl	%edx, %eax
	addl	$4, %edx
	movl	(%eax), %eax
	addl	$_stab.2650, %eax
L7:
	movl	%eax, -4(%ecx)
	addl	$4, %edx
	subl	$4, %ecx
L8:
	movl	8(%edx), %eax
	leal	(%ecx,%eax,4), %eax
	movl	-24(%eax), %edx
	movl	%edx, 20(%esp)
	movl	-20(%eax), %edx
	movl	%edx, 16(%esp)
	movl	-16(%eax), %edx
	movl	%edx, 12(%esp)
	movl	-12(%eax), %edx
	movl	%edx, 8(%esp)
	movl	-8(%eax), %edx
	movl	%edx, 4(%esp)
	movl	-4(%eax), %eax
	movl	%eax, (%esp)
	call	_printf
	movl	%esi, %edx
	movl	(%esi), %esi
	leal	8(%edx), %ecx
	movl	4(%edx), %edx
	testl	%edx, %edx
	jne	L16
L13:
	addl	$36, %esp
	xorl	%eax, %eax
	popl	%ebx
	popl	%esi
	ret
	.p2align 4,,10
L9:
	jmp	L11
	.section	.text.unlikely,"x"
LCOLDE0:
	.text
LHOTE0:
	.def	___main;	.scl	2;	.type	32;	.endef
	.section	.text.unlikely,"x"
LCOLDB1:
	.section	.text.startup,"x"
LHOTB1:
	.p2align 4,,15
	.globl	_main
	.def	_main;	.scl	2;	.type	32;	.endef
_main:
	pushl	%ebp
	movl	%esp, %ebp
	andl	$-16, %esp
	call	___main
	call	_run
	xorl	%eax, %eax
	leave
	ret
	.section	.text.unlikely,"x"
LCOLDE1:
	.section	.text.startup,"x"
LHOTE1:
	.data
	.align 4
_stab.2650:
	.byte	104
	.byte	101
	.byte	108
	.byte	108
	.byte	111
	.byte	33
	.byte	0
	.byte	0
	.section .rdata,"dr"
	.align 32
__pc.2653:
	.long	0
	.long	L4-L2
	.long	L5-L2
	.long	L6-L2
	.long	L7-L2
	.long	L8-L2
	.long	L9-L2
	.long	L10-L2
	.long	L11-L2
.lcomm _stack.2652,40000,32
	.data
	.align 32
_code.2649:
	.long	8
	.long	0
	.long	3
	.long	0
	.long	15
	.long	36
	.long	9
	.long	1
	.long	10
	.ident	"GCC: (tdm-1) 5.1.0"
	.def	_printf;	.scl	2;	.type	32;	.endef
