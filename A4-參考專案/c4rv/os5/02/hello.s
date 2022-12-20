	.file	"hello.c"
	.option nopic
	.attribute arch, "rv32i2p0_m2p0_a2p0_f2p0_d2p0"
	.attribute unaligned_access, 0
	.attribute stack_align, 16
	.text
	.section	.sdata,"aw"
	.align	2
	.type	uart, @object
	.size	uart, 4
uart:
	.word	268500992
	.section	.rodata
	.align	2
.LC0:
	.string	"Hello.\n"
	.text
	.align	2
	.globl	main
	.hidden	main
	.type	main, @function
main:
	addi	sp,sp,-32
	sw	ra,28(sp)
	sw	s0,24(sp)
	addi	s0,sp,32
	lla	a5,.LC0
	sw	a5,-20(s0)
	j	.L2
.L3:
	lw	a5,-20(s0)
	addi	a4,a5,1
	sw	a4,-20(s0)
	lbu	a5,0(a5)
	mv	a0,a5
	call	putchar
.L2:
	lw	a5,-20(s0)
	lbu	a5,0(a5)
	bne	a5,zero,.L3
.L4:
	j	.L4
	.size	main, .-main
	.align	2
	.globl	putchar
	.hidden	putchar
	.type	putchar, @function
putchar:
	addi	sp,sp,-32
	sw	s0,28(sp)
	addi	s0,sp,32
	sw	a0,-20(s0)
	nop
.L6:
	lla	a5,uart
	lw	a5,0(a5)
	lw	a5,0(a5)
	blt	a5,zero,.L6
	lla	a5,uart
	lw	a4,0(a5)
	lw	a5,-20(s0)
	andi	a5,a5,255
	sw	a5,0(a4)
	mv	a0,a5
	lw	s0,28(sp)
	addi	sp,sp,32
	jr	ra
	.size	putchar, .-putchar
	.ident	"GCC: (SiFive GCC 8.3.0-2020.04.1) 8.3.0"
