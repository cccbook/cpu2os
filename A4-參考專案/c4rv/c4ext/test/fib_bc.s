	.file	"fib_bc.c"
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
	movl	$_code.2649+216, %edx
	subl	$36, %esp
	jmp	*%ebx
	.p2align 4,,10
L76:
	.p2align 4,,10
L77:
	movl	%esi, %edx
	movl	(%esi), %esi
	leal	8(%edx), %ecx
	movl	4(%edx), %edx
	testl	%edx, %edx
	je	L101
L94:
	movl	%edx, %ebx
	subl	$_code.2649, %ebx
	sarl	$2, %ebx
	movl	__pc.2653(,%ebx,4), %ebx
	jmp	*%ebx
	.p2align 4,,10
L2:
	addl	$4, %edx
L62:
	movl	%edx, %eax
	movl	%esi, -4(%ecx)
	leal	-4(%ecx), %ebx
	movl	(%eax), %eax
	addl	$4, %edx
	movl	%ebx, %ecx
	movl	%ebx, %esi
	sall	$2, %eax
	subl	%eax, %ecx
L63:
	addl	$4, %edx
L64:
	movl	%edx, %eax
	addl	$4, %edx
	movl	(%eax), %eax
	addl	$_stab.2650, %eax
L65:
	movl	%eax, -4(%ecx)
	addl	$4, %edx
	subl	$4, %ecx
L66:
	addl	$4, %edx
L67:
	movl	%edx, %eax
	addl	$4, %edx
	movl	(%eax), %eax
L68:
	movl	%eax, -4(%ecx)
	addl	$4, %edx
	subl	$4, %ecx
L69:
	addl	$4, %edx
L70:
	leal	4(%edx), %ebx
	subl	$4, %ecx
	movl	%ebx, (%ecx)
L95:
	movl	(%edx), %ebx
	leal	(%edx,%ebx,4), %edx
	jmp	L94
	.p2align 4,,10
L44:
	addl	$4, %edx
L45:
	movl	%edx, %ebx
	addl	$4, %edx
	movl	(%ebx), %ebx
	leal	(%ecx,%ebx,4), %ecx
L46:
	movl	%eax, -4(%ecx)
	addl	$4, %edx
	subl	$4, %ecx
L47:
	addl	$4, %edx
L48:
	movl	%edx, %eax
	addl	$4, %edx
	movl	(%eax), %eax
	leal	(%esi,%eax,4), %eax
L49:
	movl	(%eax), %eax
	addl	$4, %edx
L50:
	movl	%eax, -4(%ecx)
	addl	$4, %edx
	subl	$4, %ecx
L51:
	addl	$4, %edx
L52:
	movl	%edx, %eax
	addl	$4, %edx
	movl	(%eax), %eax
L53:
	movl	%ecx, %ebx
	addl	$4, %edx
	addl	$4, %ecx
	movl	(%ebx), %ebx
	subl	%eax, %ebx
	movl	%ebx, %eax
L54:
	movl	%eax, -4(%ecx)
	addl	$4, %edx
	subl	$4, %ecx
L55:
	addl	$4, %edx
	jmp	L70
	.p2align 4,,10
L4:
	addl	$4, %edx
L5:
	movl	%edx, %eax
	movl	%esi, -4(%ecx)
	leal	-4(%ecx), %ebx
	movl	(%eax), %eax
	addl	$4, %edx
	movl	%ebx, %ecx
	movl	%ebx, %esi
	sall	$2, %eax
	subl	%eax, %ecx
L6:
	addl	$4, %edx
L7:
	movl	%edx, %eax
	addl	$4, %edx
	movl	(%eax), %eax
	leal	(%esi,%eax,4), %eax
L8:
	movl	(%eax), %eax
	addl	$4, %edx
L9:
	movl	%eax, -4(%ecx)
	addl	$4, %edx
	subl	$4, %ecx
L10:
	addl	$4, %edx
L11:
	movl	%edx, %eax
	addl	$4, %edx
	movl	(%eax), %eax
L12:
	movl	%ecx, %ebx
	addl	$4, %edx
	addl	$4, %ecx
	cmpl	(%ebx), %eax
	setge	%al
	movzbl	%al, %eax
L13:
	addl	$4, %edx
L14:
	testl	%eax, %eax
	je	L95
	addl	$4, %edx
L17:
	addl	$4, %edx
L18:
	movl	(%edx), %eax
	movl	%esi, %edx
	movl	(%esi), %esi
	leal	8(%edx), %ecx
	movl	4(%edx), %edx
	testl	%edx, %edx
	jne	L94
L101:
	addl	$36, %esp
	xorl	%eax, %eax
	popl	%ebx
	popl	%esi
	ret
	.p2align 4,,10
L21:
	addl	$4, %edx
L22:
	movl	%edx, %eax
	addl	$4, %edx
	movl	(%eax), %eax
	leal	(%esi,%eax,4), %eax
L23:
	movl	(%eax), %eax
	addl	$4, %edx
L24:
	movl	%eax, -4(%ecx)
	addl	$4, %edx
	subl	$4, %ecx
L25:
	addl	$4, %edx
L26:
	movl	%edx, %eax
	addl	$4, %edx
	movl	(%eax), %eax
L27:
	movl	%ecx, %ebx
	addl	$4, %edx
	addl	$4, %ecx
	cmpl	%eax, (%ebx)
	sete	%al
	movzbl	%al, %eax
L28:
	addl	$4, %edx
L29:
	testl	%eax, %eax
	je	L95
	addl	$4, %edx
L31:
	addl	$4, %edx
L32:
	movl	(%edx), %eax
	jmp	L77
	.p2align 4,,10
L34:
	addl	$4, %edx
L35:
	movl	%edx, %eax
	addl	$4, %edx
	movl	(%eax), %eax
	leal	(%esi,%eax,4), %eax
L36:
	movl	(%eax), %eax
	addl	$4, %edx
L37:
	movl	%eax, -4(%ecx)
	addl	$4, %edx
	subl	$4, %ecx
L38:
	addl	$4, %edx
L39:
	movl	%edx, %eax
	addl	$4, %edx
	movl	(%eax), %eax
L40:
	movl	%ecx, %ebx
	addl	$4, %edx
	addl	$4, %ecx
	movl	(%ebx), %ebx
	subl	%eax, %ebx
	movl	%ebx, %eax
L41:
	movl	%eax, -4(%ecx)
	addl	$4, %edx
	subl	$4, %ecx
L42:
	addl	$4, %edx
	jmp	L70
	.p2align 4,,10
L71:
	addl	$4, %edx
L72:
	movl	%edx, %ebx
	addl	$4, %edx
	movl	(%ebx), %ebx
	leal	(%ecx,%ebx,4), %ecx
L73:
	movl	%eax, -4(%ecx)
	addl	$4, %edx
	subl	$4, %ecx
L74:
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
	jmp	L77
	.p2align 4,,10
L57:
	addl	$4, %edx
L58:
	movl	(%edx), %edx
	leal	(%ecx,%edx,4), %ecx
L59:
	addl	(%ecx), %eax
	jmp	L77
	.p2align 4,,10
L61:
	jmp	L77
	.p2align 4,,10
L75:
	jmp	L77
	.p2align 4,,10
L43:
	jmp	L70
	.p2align 4,,10
L56:
	jmp	L70
	.p2align 4,,10
L60:
	jmp	L77
	.p2align 4,,10
L19:
	jmp	L77
	.p2align 4,,10
L33:
	jmp	L77
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
	.byte	102
	.byte	40
	.byte	55
	.byte	41
	.byte	61
	.byte	37
	.byte	100
	.byte	10
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.section .rdata,"dr"
	.align 32
__pc.2653:
	.long	L4
	.long	L5
	.long	L6
	.long	L7
	.long	L8
	.long	L9
	.long	L10
	.long	L11
	.long	L12
	.long	L13
	.long	L14
	.long	L17
	.long	L18
	.long	L19
	.long	L21
	.long	L22
	.long	L23
	.long	L24
	.long	L25
	.long	L26
	.long	L27
	.long	L28
	.long	L29
	.long	L31
	.long	L32
	.long	L33
	.long	L34
	.long	L35
	.long	L36
	.long	L37
	.long	L38
	.long	L39
	.long	L40
	.long	L41
	.long	L42
	.long	L43
	.long	L44
	.long	L45
	.long	L46
	.long	L47
	.long	L48
	.long	L49
	.long	L50
	.long	L51
	.long	L52
	.long	L53
	.long	L54
	.long	L55
	.long	L56
	.long	L57
	.long	L58
	.long	L59
	.long	L60
	.long	L61
	.long	L2
	.long	L62
	.long	L63
	.long	L64
	.long	L65
	.long	L66
	.long	L67
	.long	L68
	.long	L69
	.long	L70
	.long	L71
	.long	L72
	.long	L73
	.long	L74
	.long	L75
	.long	L76
	.long	L77
.lcomm _stack.2652,40000,32
	.data
	.align 32
_code.2649:
	.long	8
	.long	0
	.long	1
	.long	2
	.long	11
	.long	15
	.long	0
	.long	0
	.long	23
	.long	6
	.long	4
	.long	0
	.long	0
	.long	10
	.long	1
	.long	2
	.long	11
	.long	15
	.long	0
	.long	1
	.long	19
	.long	6
	.long	4
	.long	0
	.long	1
	.long	10
	.long	1
	.long	2
	.long	11
	.long	15
	.long	0
	.long	1
	.long	28
	.long	15
	.long	5
	.long	-35
	.long	9
	.long	1
	.long	15
	.long	1
	.long	2
	.long	11
	.long	15
	.long	0
	.long	2
	.long	28
	.long	15
	.long	5
	.long	-48
	.long	9
	.long	1
	.long	27
	.long	10
	.long	10
	.long	8
	.long	0
	.long	3
	.long	0
	.long	15
	.long	0
	.long	7
	.long	15
	.long	5
	.long	-63
	.long	9
	.long	1
	.long	15
	.long	36
	.long	9
	.long	2
	.long	10
	.ident	"GCC: (tdm-1) 5.1.0"
	.def	_printf;	.scl	2;	.type	32;	.endef
