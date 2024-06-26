	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 14, 0	sdk_version 14, 2
	.globl	_inc                            ; -- Begin function inc
	.p2align	2
_inc:                                   ; @inc
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #16
	.cfi_def_cfa_offset 16
	str	wzr, [sp, #12]
	b	LBB0_1
LBB0_1:                                 ; =>This Inner Loop Header: Depth=1
	ldr	w8, [sp, #12]
	mov	w9, #57600
	movk	w9, #1525, lsl #16
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB0_4
	b	LBB0_2
LBB0_2:                                 ;   in Loop: Header=BB0_1 Depth=1
	adrp	x9, _counter@PAGE
	ldr	w8, [x9, _counter@PAGEOFF]
	add	w8, w8, #1
	str	w8, [x9, _counter@PAGEOFF]
	b	LBB0_3
LBB0_3:                                 ;   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #12]
	add	w8, w8, #1
	str	w8, [sp, #12]
	b	LBB0_1
LBB0_4:
	mov	x0, #0
	add	sp, sp, #16
	ret
	.cfi_endproc
                                        ; -- End function
	.globl	_dec                            ; -- Begin function dec
	.p2align	2
_dec:                                   ; @dec
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #16
	.cfi_def_cfa_offset 16
	str	wzr, [sp, #12]
	b	LBB1_1
LBB1_1:                                 ; =>This Inner Loop Header: Depth=1
	ldr	w8, [sp, #12]
	mov	w9, #57600
	movk	w9, #1525, lsl #16
	subs	w8, w8, w9
	cset	w8, ge
	tbnz	w8, #0, LBB1_4
	b	LBB1_2
LBB1_2:                                 ;   in Loop: Header=BB1_1 Depth=1
	adrp	x9, _counter@PAGE
	ldr	w8, [x9, _counter@PAGEOFF]
	subs	w8, w8, #1
	str	w8, [x9, _counter@PAGEOFF]
	b	LBB1_3
LBB1_3:                                 ;   in Loop: Header=BB1_1 Depth=1
	ldr	w8, [sp, #12]
	add	w8, w8, #1
	str	w8, [sp, #12]
	b	LBB1_1
LBB1_4:
	mov	x0, #0
	add	sp, sp, #16
	ret
	.cfi_endproc
                                        ; -- End function
	.globl	_main                           ; -- Begin function main
	.p2align	2
_main:                                  ; @main
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #48
	.cfi_def_cfa_offset 48
	stp	x29, x30, [sp, #32]             ; 16-byte Folded Spill
	add	x29, sp, #32
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	sub	x0, x29, #8
	mov	x3, #0
	str	x3, [sp, #8]                    ; 8-byte Folded Spill
	mov	x1, x3
	adrp	x2, _inc@PAGE
	add	x2, x2, _inc@PAGEOFF
	bl	_pthread_create
	ldr	x3, [sp, #8]                    ; 8-byte Folded Reload
	add	x0, sp, #16
	mov	x1, x3
	adrp	x2, _dec@PAGE
	add	x2, x2, _dec@PAGEOFF
	bl	_pthread_create
	ldr	x1, [sp, #8]                    ; 8-byte Folded Reload
	ldur	x0, [x29, #-8]
	bl	_pthread_join
	ldr	x1, [sp, #8]                    ; 8-byte Folded Reload
	ldr	x0, [sp, #16]
	bl	_pthread_join
	adrp	x8, _counter@PAGE
	ldr	w9, [x8, _counter@PAGEOFF]
                                        ; implicit-def: $x8
	mov	x8, x9
	mov	x9, sp
	str	x8, [x9]
	adrp	x0, l_.str@PAGE
	add	x0, x0, l_.str@PAGEOFF
	bl	_printf
	mov	w0, #0
	ldp	x29, x30, [sp, #32]             ; 16-byte Folded Reload
	add	sp, sp, #48
	ret
	.cfi_endproc
                                        ; -- End function
	.globl	_counter                        ; @counter
.zerofill __DATA,__common,_counter,4,2
	.section	__TEXT,__cstring,cstring_literals
l_.str:                                 ; @.str
	.asciz	"counter=%d\n"

.subsections_via_symbols
