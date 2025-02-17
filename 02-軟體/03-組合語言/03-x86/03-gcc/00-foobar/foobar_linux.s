	.file	"foobar.c"
# GNU C11 (Ubuntu 7.5.0-3ubuntu1~18.04) version 7.5.0 (x86_64-linux-gnu)
#	compiled by GNU C version 7.5.0, GMP version 6.1.2, MPFR version 4.0.1, MPC version 1.1.0, isl version isl-0.19-GMP

# GGC heuristics: --param ggc-min-expand=97 --param ggc-min-heapsize=126130
# options passed:  -imultiarch x86_64-linux-gnu foobar.c -mtune=generic
# -march=x86-64 -auxbase-strip foobar_linux.s -fverbose-asm
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
# -flto-odr-type-merging -fmath-errno -fmerge-debug-strings -fpeephole
# -fplt -fprefetch-loop-arrays -freg-struct-return
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
# -funwind-tables -fverbose-asm -fzero-initialized-in-bss
# -m128bit-long-double -m64 -m80387 -malign-stringops
# -mavx256-split-unaligned-load -mavx256-split-unaligned-store
# -mfancy-math-387 -mfp-ret-in-387 -mfxsr -mglibc -mieee-fp
# -mlong-double-80 -mmmx -mno-sse4 -mpush-args -mred-zone -msse -msse2
# -mstv -mtls-direct-seg-refs -mvzeroupper

	.text
	.globl	foobar
	.type	foobar, @function
foobar:
.LFB0:
	.cfi_startproc
	pushq	%rbp	#
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp	#,
	.cfi_def_cfa_register 6
	movl	%edi, -20(%rbp)	# a, a
	movl	%esi, -24(%rbp)	# b, b
	movl	%edx, -28(%rbp)	# c, c
# foobar.c:5:     int xx = a + 2;
	movl	-20(%rbp), %eax	# a, tmp95
	addl	$2, %eax	#, tmp94
	movl	%eax, -16(%rbp)	# tmp94, xx
# foobar.c:6:     int yy = b + 3;
	movl	-24(%rbp), %eax	# b, tmp99
	addl	$3, %eax	#, tmp98
	movl	%eax, -12(%rbp)	# tmp98, yy
# foobar.c:7:     int zz = c + 4;
	movl	-28(%rbp), %eax	# c, tmp103
	addl	$4, %eax	#, tmp102
	movl	%eax, -8(%rbp)	# tmp102, zz
# foobar.c:8:     int sum = xx + yy + zz;
	movl	-16(%rbp), %edx	# xx, tmp104
	movl	-12(%rbp), %eax	# yy, tmp105
	addl	%eax, %edx	# tmp105, _1
	movl	-8(%rbp), %eax	# zz, tmp109
	addl	%edx, %eax	# _1, tmp108
	movl	%eax, -4(%rbp)	# tmp108, sum
# foobar.c:10:     return xx * yy * zz + sum;
	movl	-16(%rbp), %eax	# xx, tmp110
	imull	-12(%rbp), %eax	# yy, _2
	imull	-8(%rbp), %eax	# zz, _2
	movl	%eax, %edx	# _2, _3
	movl	-4(%rbp), %eax	# sum, tmp111
	addl	%edx, %eax	# _3, _11
# foobar.c:11: }
	popq	%rbp	#
	.cfi_def_cfa 7, 8
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
	pushq	%rbp	#
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp	#,
	.cfi_def_cfa_register 6
	subq	$16, %rsp	#,
# foobar.c:15:     int r = foobar(77, 88, 99);
	movl	$99, %edx	#,
	movl	$88, %esi	#,
	movl	$77, %edi	#,
	call	foobar	#
	movl	%eax, -4(%rbp)	# tmp89, r
# foobar.c:16:     printf("r=%d\n", r);
	movl	-4(%rbp), %eax	# r, tmp90
	movl	%eax, %esi	# tmp90,
	leaq	.LC0(%rip), %rdi	#,
	movl	$0, %eax	#,
	call	printf@PLT	#
	movl	$0, %eax	#, _5
# foobar.c:17: }
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	main, .-main
	.ident	"GCC: (Ubuntu 7.5.0-3ubuntu1~18.04) 7.5.0"
	.section	.note.GNU-stack,"",@progbits
