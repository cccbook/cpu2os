	.file	"sum_bc.c"
	.section	.text.unlikely,"x"
LCOLDB0:
	.text
LHOTB0:
	.p2align 4,,15
	.globl	_run
	.def	_run;	.scl	2;	.type	32;	.endef
_run:
	pushl	%edi
	pushl	%esi
	movl	$_stack.2652+40000, %edi
	pushl	%ebx
	movl	$L2, %edx
	movl	%edi, %esi
	movl	$_code.2649+336, %ebx
	subl	$32, %esp
	jmp	*%edx
L105:
	.p2align 4,,10
L106:
	movl	%edi, %edx
	movl	(%edi), %edi
	movl	4(%edx), %ebx
	leal	8(%edx), %esi
	testl	%ebx, %ebx
	je	L122
	.p2align 4,,10
L117:
	movl	%ebx, %edx
	subl	$_code.2649, %edx
	sarl	$2, %edx
	movl	__pc.2653(,%edx,4), %edx
	jmp	*%edx
L4:
	addl	$4, %ebx
L5:
	movl	%ebx, %eax
	movl	%edi, -4(%esi)
	leal	-4(%esi), %edx
	movl	(%eax), %eax
	addl	$4, %ebx
	movl	%edx, %esi
	movl	%edx, %edi
	sall	$2, %eax
	subl	%eax, %esi
L6:
	addl	$4, %ebx
L7:
	movl	%ebx, %eax
	addl	$4, %ebx
	movl	(%eax), %eax
	leal	(%edi,%eax,4), %eax
L8:
	movl	%eax, -4(%esi)
	addl	$4, %ebx
	subl	$4, %esi
L9:
	addl	$4, %ebx
L10:
	movl	%ebx, %eax
	addl	$4, %ebx
	movl	(%eax), %eax
L11:
	movl	%esi, %edx
	addl	$4, %ebx
	addl	$4, %esi
	movl	(%edx), %edx
	movl	%eax, (%edx)
L12:
	addl	$4, %ebx
L13:
	movl	%ebx, %eax
	addl	$4, %ebx
	movl	(%eax), %eax
	leal	(%edi,%eax,4), %eax
L14:
	movl	%eax, -4(%esi)
	addl	$4, %ebx
	subl	$4, %esi
L15:
	addl	$4, %ebx
L16:
	movl	%ebx, %eax
	addl	$4, %ebx
	movl	(%eax), %eax
L17:
	movl	%esi, %edx
	addl	$4, %ebx
	addl	$4, %esi
	movl	(%edx), %edx
	movl	%eax, (%edx)
L18:
	addl	$4, %ebx
L19:
	movl	%ebx, %eax
	addl	$4, %ebx
	movl	(%eax), %eax
	addl	$_stab.2650, %eax
L20:
	movl	%eax, -4(%esi)
	addl	$4, %ebx
	subl	$4, %esi
L21:
	addl	$4, %ebx
L22:
	movl	%ebx, %eax
	addl	$4, %ebx
	movl	(%eax), %eax
	leal	(%edi,%eax,4), %eax
L23:
	movl	(%eax), %eax
	addl	$4, %ebx
L24:
	movl	%eax, -4(%esi)
	addl	$4, %ebx
	subl	$4, %esi
L25:
	addl	$4, %ebx
L26:
	movl	%ebx, %eax
	addl	$4, %ebx
	movl	(%eax), %eax
	leal	(%edi,%eax,4), %eax
L27:
	movl	(%eax), %eax
	addl	$4, %ebx
L28:
	movl	%eax, -4(%esi)
	addl	$4, %ebx
	subl	$4, %esi
L29:
	movl	8(%ebx), %eax
	addl	$4, %ebx
	leal	(%esi,%eax,4), %eax
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
L30:
	addl	$4, %ebx
L31:
	movl	%ebx, %eax
	addl	$4, %ebx
	movl	(%eax), %eax
	leal	(%esi,%eax,4), %esi
L32:
	addl	$4, %ebx
L33:
	movl	%ebx, %eax
	addl	$4, %ebx
	movl	(%eax), %eax
	leal	(%edi,%eax,4), %eax
L34:
	movl	(%eax), %eax
	addl	$4, %ebx
L35:
	movl	%eax, -4(%esi)
	addl	$4, %ebx
	subl	$4, %esi
L36:
	addl	$4, %ebx
L37:
	movl	%ebx, %eax
	addl	$4, %ebx
	movl	(%eax), %eax
	leal	(%edi,%eax,4), %eax
L38:
	movl	(%eax), %eax
	addl	$4, %ebx
L39:
	movl	%esi, %edx
	addl	$4, %ebx
	addl	$4, %esi
	cmpl	(%edx), %eax
	setge	%al
	movzbl	%al, %eax
L40:
	addl	$4, %ebx
L41:
	testl	%eax, %eax
	je	L118
	addl	$4, %ebx
L44:
	addl	$4, %ebx
L45:
	movl	%ebx, %eax
	addl	$4, %ebx
	movl	(%eax), %eax
	addl	$_stab.2650, %eax
L46:
	movl	%eax, -4(%esi)
	addl	$4, %ebx
	subl	$4, %esi
L47:
	addl	$4, %ebx
L48:
	movl	%ebx, %eax
	addl	$4, %ebx
	movl	(%eax), %eax
	leal	(%edi,%eax,4), %eax
L49:
	movl	(%eax), %eax
	addl	$4, %ebx
L50:
	movl	%eax, -4(%esi)
	addl	$4, %ebx
	subl	$4, %esi
L51:
	addl	$4, %ebx
L52:
	movl	%ebx, %eax
	addl	$4, %ebx
	movl	(%eax), %eax
	leal	(%edi,%eax,4), %eax
L53:
	movl	(%eax), %eax
	addl	$4, %ebx
L54:
	movl	%eax, -4(%esi)
	addl	$4, %ebx
	subl	$4, %esi
L55:
	movl	8(%ebx), %eax
	addl	$4, %ebx
	leal	(%esi,%eax,4), %eax
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
L56:
	addl	$4, %ebx
L57:
	movl	%ebx, %eax
	addl	$4, %ebx
	movl	(%eax), %eax
	leal	(%esi,%eax,4), %esi
L58:
	addl	$4, %ebx
L59:
	movl	%ebx, %eax
	addl	$4, %ebx
	movl	(%eax), %eax
	leal	(%edi,%eax,4), %eax
L60:
	movl	%eax, -4(%esi)
	addl	$4, %ebx
	subl	$4, %esi
L61:
	addl	$4, %ebx
L62:
	movl	%ebx, %eax
	addl	$4, %ebx
	movl	(%eax), %eax
	leal	(%edi,%eax,4), %eax
L63:
	movl	(%eax), %eax
	addl	$4, %ebx
L64:
	movl	%eax, -4(%esi)
	addl	$4, %ebx
	subl	$4, %esi
L65:
	addl	$4, %ebx
L66:
	movl	%ebx, %eax
	addl	$4, %ebx
	movl	(%eax), %eax
	leal	(%edi,%eax,4), %eax
L67:
	movl	(%eax), %eax
	addl	$4, %ebx
L68:
	movl	%esi, %edx
	addl	$4, %ebx
	addl	$4, %esi
	addl	(%edx), %eax
L69:
	movl	%esi, %edx
	addl	$4, %ebx
	addl	$4, %esi
	movl	(%edx), %edx
	movl	%eax, (%edx)
L70:
	addl	$4, %ebx
L71:
	movl	%ebx, %eax
	addl	$4, %ebx
	movl	(%eax), %eax
	leal	(%edi,%eax,4), %eax
L72:
	movl	%eax, -4(%esi)
	addl	$4, %ebx
	subl	$4, %esi
L73:
	movl	(%eax), %eax
	addl	$4, %ebx
L74:
	movl	%eax, -4(%esi)
	addl	$4, %ebx
	subl	$4, %esi
L75:
	addl	$4, %ebx
L76:
	movl	%ebx, %eax
	addl	$4, %ebx
	movl	(%eax), %eax
L77:
	movl	%esi, %edx
	addl	$4, %ebx
	addl	$4, %esi
	addl	(%edx), %eax
L78:
	movl	%esi, %edx
	addl	$4, %ebx
	addl	$4, %esi
	movl	(%edx), %edx
	movl	%eax, (%edx)
L79:
	movl	%eax, -4(%esi)
	addl	$4, %ebx
	subl	$4, %esi
L80:
	addl	$4, %ebx
L81:
	movl	%ebx, %eax
	addl	$4, %ebx
	movl	(%eax), %eax
L82:
	movl	%esi, %edx
	addl	$4, %ebx
	addl	$4, %esi
	movl	(%edx), %edx
	subl	%eax, %edx
	movl	%edx, %eax
L83:
	addl	$4, %ebx
L118:
	movl	(%ebx), %edx
	leal	(%ebx,%edx,4), %ebx
	jmp	L117
L2:
	addl	$4, %ebx
L91:
	movl	%ebx, %eax
	movl	%edi, -4(%esi)
	leal	-4(%esi), %edx
	movl	(%eax), %eax
	addl	$4, %ebx
	movl	%edx, %esi
	movl	%edx, %edi
	sall	$2, %eax
	subl	%eax, %esi
L92:
	addl	$4, %ebx
L93:
	movl	%ebx, %eax
	addl	$4, %ebx
	movl	(%eax), %eax
	addl	$_stab.2650, %eax
L94:
	movl	%eax, -4(%esi)
	addl	$4, %ebx
	subl	$4, %esi
L95:
	addl	$4, %ebx
L96:
	movl	%ebx, %eax
	addl	$4, %ebx
	movl	(%eax), %eax
L97:
	movl	%eax, -4(%esi)
	addl	$4, %ebx
	subl	$4, %esi
L98:
	addl	$4, %ebx
L99:
	leal	4(%ebx), %edx
	subl	$4, %esi
	movl	%edx, (%esi)
	jmp	L118
L100:
	addl	$4, %ebx
L101:
	movl	%ebx, %edx
	addl	$4, %ebx
	movl	(%edx), %edx
	leal	(%esi,%edx,4), %esi
L102:
	movl	%eax, -4(%esi)
	addl	$4, %ebx
	subl	$4, %esi
L103:
	movl	8(%ebx), %eax
	leal	(%esi,%eax,4), %eax
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
	movl	%edi, %edx
	movl	(%edi), %edi
	movl	4(%edx), %ebx
	leal	8(%edx), %esi
	testl	%ebx, %ebx
	jne	L117
L122:
	addl	$32, %esp
	xorl	%eax, %eax
	popl	%ebx
	popl	%esi
	popl	%edi
	ret
L85:
	addl	$4, %ebx
L86:
	movl	(%ebx), %eax
	leal	(%edi,%eax,4), %eax
L87:
	movl	(%eax), %eax
	jmp	L106
L88:
	jmp	L106
L84:
	jmp	L118
L90:
	jmp	L106
L104:
	jmp	L106
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
	.align 32
_stab.2650:
	.byte	105
	.byte	61
	.byte	37
	.byte	100
	.byte	32
	.byte	115
	.byte	61
	.byte	37
	.byte	100
	.byte	10
	.byte	0
	.byte	0
	.byte	105
	.byte	61
	.byte	37
	.byte	100
	.byte	32
	.byte	115
	.byte	61
	.byte	37
	.byte	100
	.byte	10
	.byte	0
	.byte	0
	.byte	115
	.byte	117
	.byte	109
	.byte	40
	.byte	49
	.byte	48
	.byte	41
	.byte	61
	.byte	37
	.byte	100
	.byte	10
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
	.long	L15
	.long	L16
	.long	L17
	.long	L18
	.long	L19
	.long	L20
	.long	L21
	.long	L22
	.long	L23
	.long	L24
	.long	L25
	.long	L26
	.long	L27
	.long	L28
	.long	L29
	.long	L30
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
	.long	L78
	.long	L79
	.long	L80
	.long	L81
	.long	L82
	.long	L83
	.long	L84
	.long	L85
	.long	L86
	.long	L87
	.long	L88
	.long	L90
	.long	L2
	.long	L91
	.long	L92
	.long	L93
	.long	L94
	.long	L95
	.long	L96
	.long	L97
	.long	L98
	.long	L99
	.long	L100
	.long	L101
	.long	L102
	.long	L103
	.long	L104
	.long	L105
	.long	L106
.lcomm _stack.2652,40000,32
	.data
	.align 32
_code.2649:
	.long	8
	.long	2
	.long	1
	.long	-1
	.long	15
	.long	0
	.long	0
	.long	13
	.long	1
	.long	-2
	.long	15
	.long	0
	.long	1
	.long	13
	.long	3
	.long	0
	.long	15
	.long	1
	.long	-2
	.long	11
	.long	15
	.long	1
	.long	-1
	.long	11
	.long	15
	.long	36
	.long	9
	.long	3
	.long	1
	.long	-2
	.long	11
	.long	15
	.long	1
	.long	2
	.long	11
	.long	23
	.long	6
	.long	42
	.long	3
	.long	12
	.long	15
	.long	1
	.long	-2
	.long	11
	.long	15
	.long	1
	.long	-1
	.long	11
	.long	15
	.long	36
	.long	9
	.long	3
	.long	1
	.long	-1
	.long	15
	.long	1
	.long	-1
	.long	11
	.long	15
	.long	1
	.long	-2
	.long	11
	.long	27
	.long	13
	.long	1
	.long	-2
	.long	15
	.long	11
	.long	15
	.long	0
	.long	1
	.long	27
	.long	13
	.long	15
	.long	0
	.long	1
	.long	28
	.long	4
	.long	-50
	.long	1
	.long	-1
	.long	11
	.long	10
	.long	10
	.long	8
	.long	0
	.long	3
	.long	24
	.long	15
	.long	0
	.long	10
	.long	15
	.long	5
	.long	-93
	.long	9
	.long	1
	.long	15
	.long	36
	.long	9
	.long	2
	.long	10
	.ident	"GCC: (tdm-1) 5.1.0"
	.def	_printf;	.scl	2;	.type	32;	.endef
