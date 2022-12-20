	.file	"foobar.c"
# GNU C11 (Ubuntu 7.5.0-3ubuntu1~18.04) version 7.5.0 (x86_64-linux-gnu)
#	compiled by GNU C version 7.5.0, GMP version 6.1.2, MPFR version 4.0.1, MPC version 1.1.0, isl version isl-0.19-GMP

# GGC heuristics: --param ggc-min-expand=97 --param ggc-min-heapsize=126130
# options passed:  -imultilib 32 -imultiarch i386-linux-gnu foobar.c -m32
# -mtune=generic -march=i686 -auxbase-strip foobar_32.s -fverbose-asm
# -fstack-protector-strong -Wformat -Wformat-security
# options enabled:  -fPIC -fPIE -faggressive-loop-optimizations
# -fasynchronous-unwind-tables -fauto-inc-dec -fchkp-check-incomplete-type
# -fchkp-check-read -fchkp-check-write -fchkp-instrument-calls
# -fchkp-narrow-bounds -fchkp-optimize -fchkp-store-bounds
# -fchkp-use-static-bounds -fchkp-use-static-const-bounds
# -fchkp-use-wrappers -fcommon -fdelete-null-pointer-checks
# -fdwarf2-cfi-asm -fearly-inlining -feliminate-unused-debug-types
# -ffp-int-builtin-inexact -ffunction-cse -fgcse-lm -fgnu-runtime
# -fgnu-unique -fident -finline-atomics -fira-hoist-pressure
# -fira-share-save-slots -fira-share-spill-slots -fivopts
# -fkeep-static-consts -fleading-underscore -flifetime-dse
# -flto-odr-type-merging -fmath-errno -fmerge-debug-strings
# -fpcc-struct-return -fpeephole -fplt -fprefetch-loop-arrays
# -fsched-critical-path-heuristic -fsched-dep-count-heuristic
# -fsched-group-heuristic -fsched-interblock -fsched-last-insn-heuristic
# -fsched-rank-heuristic -fsched-spec -fsched-spec-insn-heuristic
# -fsched-stalled-insns-dep -fschedule-fusion -fsemantic-interposition
# -fshow-column -fshrink-wrap-separate -fsigned-zeros
# -fsplit-ivs-in-unroller -fssa-backprop -fstack-protector-strong
# -fstdarg-opt -fstrict-volatile-bitfields -fsync-libcalls -ftrapping-math
# -ftree-cselim -ftree-forwprop -ftree-loop-if-convert -ftree-loop-im
# -ftree-loop-ivcanon -ftree-loop-optimize -ftree-parallelize-loops=
# -ftree-phiprop -ftree-reassoc -ftree-scev-cprop -funit-at-a-time
# -funwind-tables -fverbose-asm -fzero-initialized-in-bss -m32 -m80387
# -m96bit-long-double -malign-stringops -mavx256-split-unaligned-load
# -mavx256-split-unaligned-store -mfancy-math-387 -mfp-ret-in-387 -mglibc
# -mieee-fp -mlong-double-80 -mno-red-zone -mno-sse4 -mpush-args -msahf
# -mstv -mtls-direct-seg-refs -mvzeroupper

	.text
	.globl	foobar
	.type	foobar, @function
foobar:
.LFB0:
	.cfi_startproc
	pushl	%ebp	#
	.cfi_def_cfa_offset 8
	.cfi_offset 5, -8
	movl	%esp, %ebp	#,
	.cfi_def_cfa_register 5
	subl	$16, %esp	#,
	call	__x86.get_pc_thunk.ax	#
	addl	$_GLOBAL_OFFSET_TABLE_, %eax	# tmp87,
# foobar.c:5:     int xx = a + 2;
	movl	8(%ebp), %eax	# a, tmp96
	addl	$2, %eax	#, tmp95
	movl	%eax, -16(%ebp)	# tmp95, xx
# foobar.c:6:     int yy = b + 3;
	movl	12(%ebp), %eax	# b, tmp100
	addl	$3, %eax	#, tmp99
	movl	%eax, -12(%ebp)	# tmp99, yy
# foobar.c:7:     int zz = c + 4;
	movl	16(%ebp), %eax	# c, tmp104
	addl	$4, %eax	#, tmp103
	movl	%eax, -8(%ebp)	# tmp103, zz
# foobar.c:8:     int sum = xx + yy + zz;
	movl	-16(%ebp), %edx	# xx, tmp105
	movl	-12(%ebp), %eax	# yy, tmp106
	addl	%eax, %edx	# tmp106, _1
	movl	-8(%ebp), %eax	# zz, tmp110
	addl	%edx, %eax	# _1, tmp109
	movl	%eax, -4(%ebp)	# tmp109, sum
# foobar.c:10:     return xx * yy * zz + sum;
	movl	-16(%ebp), %eax	# xx, tmp111
	imull	-12(%ebp), %eax	# yy, _2
	imull	-8(%ebp), %eax	# zz, _2
	movl	%eax, %edx	# _2, _3
	movl	-4(%ebp), %eax	# sum, tmp112
	addl	%edx, %eax	# _3, _11
# foobar.c:11: }
	leave
	.cfi_restore 5
	.cfi_def_cfa 4, 4
	ret
	.cfi_endproc
.LFE0:
	.size	foobar, .-foobar
	.section	.rodata
.LC0:
	.string	"r=%d\n"
	.text
	.globl	main
	.type	main, @function
main:
.LFB1:
	.cfi_startproc
	leal	4(%esp), %ecx	#,
	.cfi_def_cfa 1, 0
	andl	$-16, %esp	#,
	pushl	-4(%ecx)	#
	pushl	%ebp	#
	.cfi_escape 0x10,0x5,0x2,0x75,0
	movl	%esp, %ebp	#,
	pushl	%ebx	#
	pushl	%ecx	#
	.cfi_escape 0xf,0x3,0x75,0x78,0x6
	.cfi_escape 0x10,0x3,0x2,0x75,0x7c
	subl	$16, %esp	#,
	call	__x86.get_pc_thunk.bx	#
	addl	$_GLOBAL_OFFSET_TABLE_, %ebx	# tmp87,
# foobar.c:15:     int r = foobar(77, 88, 99);
	pushl	$99	#
	pushl	$88	#
	pushl	$77	#
	call	foobar	#
	addl	$12, %esp	#,
	movl	%eax, -12(%ebp)	# tmp90, r
# foobar.c:16:     printf("r=%d\n", r);
	subl	$8, %esp	#,
	pushl	-12(%ebp)	# r
	leal	.LC0@GOTOFF(%ebx), %eax	#, tmp91
	pushl	%eax	# tmp91
	call	printf@PLT	#
	addl	$16, %esp	#,
	movl	$0, %eax	#, _5
# foobar.c:17: }
	leal	-8(%ebp), %esp	#,
	popl	%ecx	#
	.cfi_restore 1
	.cfi_def_cfa 1, 0
	popl	%ebx	#
	.cfi_restore 3
	popl	%ebp	#
	.cfi_restore 5
	leal	-4(%ecx), %esp	#,
	.cfi_def_cfa 4, 4
	ret
	.cfi_endproc
.LFE1:
	.size	main, .-main
	.section	.text.__x86.get_pc_thunk.ax,"axG",@progbits,__x86.get_pc_thunk.ax,comdat
	.globl	__x86.get_pc_thunk.ax
	.hidden	__x86.get_pc_thunk.ax
	.type	__x86.get_pc_thunk.ax, @function
__x86.get_pc_thunk.ax:
.LFB2:
	.cfi_startproc
	movl	(%esp), %eax	#,
	ret
	.cfi_endproc
.LFE2:
	.section	.text.__x86.get_pc_thunk.bx,"axG",@progbits,__x86.get_pc_thunk.bx,comdat
	.globl	__x86.get_pc_thunk.bx
	.hidden	__x86.get_pc_thunk.bx
	.type	__x86.get_pc_thunk.bx, @function
__x86.get_pc_thunk.bx:
.LFB3:
	.cfi_startproc
	movl	(%esp), %ebx	#,
	ret
	.cfi_endproc
.LFE3:
	.ident	"GCC: (Ubuntu 7.5.0-3ubuntu1~18.04) 7.5.0"
	.section	.note.GNU-stack,"",@progbits
