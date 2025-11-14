	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 15, 0	sdk_version 15, 2
	.globl	_add				; -- Begin function add
	.p2align	2
_add:					; @add
	.cfi_startproc
	sub	sp, sp, #32
	.cfi_def_cfa_offset 32
	stp	x29, x30, [sp, #16]
	str	w0, [sp, #12]		; Store param 'a'
	str	w1, [sp, #8]		; Store param 'b'
	ldr	w8, [sp, #-4]		; Load var 'a'
	ldr	w9, [sp, #-8]		; Load var 'b'
	add	w10, w8, w9
	str	w10, [sp, #-20]		; Store var 'sum'
	ldr	w11, [sp, #-20]		; Load var 'sum'
	mov	w0, w11			; Set return value from t3
	ldp	x29, x30, [sp, #16]
	add	sp, sp, #32
	ret
	.cfi_endproc
						; -- End function add
	.globl	_main				; -- Begin function main
	.p2align	2
_main:					; @main
	.cfi_startproc
	sub	sp, sp, #32
	.cfi_def_cfa_offset 32
	stp	x29, x30, [sp, #16]
	mov	w8, #10
	str	w8, [sp, #-20]		; Store var 'x'
	ldr	w9, [sp, #-20]		; Load var 'x'
	mov	w0, w9			; Set arg0
	mov	w10, #20
	mov	w1, w10			; Set arg1
	bl	_add
	mov	w11, w0			; Get return value
	str	w11, [sp, #-24]		; Store var 'y'
	ldr	w12, [sp, #-24]		; Load var 'y'
	mov	w0, w12			; Set arg0
	mov	w13, #5
	mov	w1, w13			; Set arg1
	bl	_add
	mov	w14, w0			; Get return value
	str	w14, [sp, #-28]		; Store var 'z'
	ldr	w15, [sp, #-28]		; Load var 'z'
	mov	w0, w15			; Set return value from t11
	ldp	x29, x30, [sp, #16]
	add	sp, sp, #32
	ret
	.cfi_endproc
						; -- End function main
.subsections_via_symbols
