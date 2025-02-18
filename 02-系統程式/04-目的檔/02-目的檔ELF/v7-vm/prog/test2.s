
	.text
	.globl	main
	.type	main, @function
main:
	addi x1, x0, 2
	addi x2, x1, 3   
	sw   x2, 100(x0)
    ret
