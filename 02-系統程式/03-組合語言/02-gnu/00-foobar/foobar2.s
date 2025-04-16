	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 14, 0	sdk_version 14, 2
	.globl	_foobar                         ; -- Begin function foobar
	.p2align	2
_foobar:                                ; @foobar
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #32
	.cfi_def_cfa_offset 32
	str	w0, [sp, #28]
	str	w1, [sp, #24]
	str	w2, [sp, #20]
	ldr	w8, [sp, #28]
	add	w8, w8, #2
	str	w8, [sp, #16]
	ldr	w8, [sp, #24]
	add	w8, w8, #3
	str	w8, [sp, #12]
	ldr	w8, [sp, #20]
	add	w8, w8, #4
	str	w8, [sp, #8]
	ldr	w8, [sp, #16]
	ldr	w9, [sp, #12]
	add	w8, w8, w9
	ldr	w9, [sp, #8]
	add	w8, w8, w9
	str	w8, [sp, #4]
	ldr	w8, [sp, #16]
	ldr	w9, [sp, #12]
	mul	w8, w8, w9
	ldr	w9, [sp, #8]
	mul	w8, w8, w9
	ldr	w9, [sp, #4]
	add	w0, w8, w9
	add	sp, sp, #32
	ret
	.cfi_endproc
                                        ; -- End function
	.globl	_main                           ; -- Begin function main
	.p2align	2
_main:                                  ; @main
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #32
	.cfi_def_cfa_offset 32
	stp	x29, x30, [sp, #16]             ; 16-byte Folded Spill
	add	x29, sp, #16
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	mov	w0, #77
	mov	w1, #88
	mov	w2, #99
	bl	_foobar
	stur	w0, [x29, #-4]
	ldur	w9, [x29, #-4]
                                        ; implicit-def: $x8
	mov	x8, x9
	mov	x9, sp
	str	x8, [x9]
	adrp	x0, l_.str@PAGE
	add	x0, x0, l_.str@PAGEOFF
	bl	_printf
	mov	w0, #0
	ldp	x29, x30, [sp, #16]             ; 16-byte Folded Reload
	add	sp, sp, #32
	ret
	.cfi_endproc
                                        ; -- End function
	.section	__TEXT,__cstring,cstring_literals
l_.str:                                 ; @.str
	.asciz	"r=%d\n"

.subsections_via_symbols
