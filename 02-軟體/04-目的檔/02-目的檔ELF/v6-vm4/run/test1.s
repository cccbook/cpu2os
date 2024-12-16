	.file	"test1.c"
	.option nopic
	.attribute arch, "rv32i2p1_m2p0_a2p1_f2p2_d2p2_zicsr2p0_zifencei2p0"
	.attribute unaligned_access, 0
	.attribute stack_align, 16
	.text
	.globl	temp
	.section	.sbss,"aw",@nobits
	.align	2
	.type	temp, @object
	.size	temp, 4
temp:
	.zero	4
	.text
	.align	2
	.globl	sum
	.type	sum, @function
sum:
	addi	sp,sp,-48
	sw	s0,44(sp)
	addi	s0,sp,48
	sw	a0,-36(s0)
	sw	zero,-20(s0)
	li	a5,1
	sw	a5,-24(s0)
	j	.L2
.L3:
	lw	a4,-20(s0)
	lw	a5,-24(s0)
	add	a5,a4,a5
	sw	a5,-20(s0)
	lw	a5,-24(s0)
	addi	a5,a5,1
	sw	a5,-24(s0)
.L2:
	lw	a4,-24(s0)
	lw	a5,-36(s0)
	ble	a4,a5,.L3
	lw	a5,-20(s0)
	mv	a0,a5
	lw	s0,44(sp)
	addi	sp,sp,48
	jr	ra
	.size	sum, .-sum
	.section	.rodata
	.align	2
.LC0:
	.string	"hello"
	.text
	.align	2
	.globl	main
	.type	main, @function
main:
	addi	sp,sp,-32
	sw	ra,28(sp)
	sw	s0,24(sp)
	addi	s0,sp,32
	li	a0,10
	call	sum
	sw	a0,-20(s0)
	lui	a5,%hi(.LC0)
	addi	a5,a5,%lo(.LC0)
 #APP
# 20 "test1.c" 1
	li a7, 7
la a6, a5
ecall

# 0 "" 2
 #NO_APP
	lui	a5,%hi(temp)
	sw	a4,%lo(temp)(a5)
	li	a5,0
	mv	a0,a5
	lw	ra,28(sp)
	lw	s0,24(sp)
	addi	sp,sp,32
	jr	ra
	.size	main, .-main
	.ident	"GCC: (gc891d8dc2) 13.2.0"
