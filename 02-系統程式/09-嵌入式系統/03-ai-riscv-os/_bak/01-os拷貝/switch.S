# switch.S: Context switching and interrupt vector

.section .text
.globl switch_context
.globl timer_vector

# ============================================================================
# void switch_context(struct context *old, struct context *new);
# a0: pointer to the old thread's context
# a1: pointer to the new thread's context
#
# This function saves the callee-saved registers of the current thread
# into its context structure and restores the registers for the next
# thread from its context structure.
# ============================================================================
switch_context:
    # Save callee-saved registers of the old context
    sd ra, 0(a0)
    sd sp, 8(a0)
    sd s0, 16(a0)
    sd s1, 24(a0)
    sd s2, 32(a0)
    sd s3, 40(a0)
    sd s4, 48(a0)
    sd s5, 56(a0)
    sd s6, 64(a0)
    sd s7, 72(a0)
    sd s8, 80(a0)
    sd s9, 88(a0)
    sd s10, 96(a0)
    sd s11, 104(a0)

    # Restore callee-saved registers of the new context
    ld ra, 0(a1)
    ld sp, 8(a1)
    ld s0, 16(a1)
    ld s1, 24(a1)
    ld s2, 32(a1)
    ld s3, 40(a1)
    ld s4, 48(a1)
    ld s5, 56(a1)
    ld s6, 64(a1)
    ld s7, 72(a1)
    ld s8, 80(a1)
    ld s9, 88(a1)
    ld s10, 96(a1)
    ld s11, 104(a1)

    ret # Return to the new context's ra

# ============================================================================
# timer_vector
#
# This is the entry point for machine-mode timer interrupts.
# It saves the entire context of the interrupted thread, calls the C
# handler (timer_handler), and then restores the context before returning.
# ============================================================================
.align 4
timer_vector:
    # 1. Save the full context on the stack
    # We need to save all general-purpose registers that might be used
    # by the C handler. There are 32 GPRs (x0-x31).
    # x0 (zero) is hardwired to 0.
    # We don't need to save sp (x2) explicitly at the beginning,
    # as we are modifying it to allocate space.

    # Allocate stack frame (31 registers * 8 bytes/register = 248 bytes)
    # Align to 256 for simplicity
    addi sp, sp, -256

    sd ra, 0(sp)   # x1
    # sp (x2) is our stack pointer
    sd gp, 16(sp)  # x3
    sd tp, 24(sp)  # x4
    sd t0, 32(sp)  # x5
    sd t1, 40(sp)  # x6
    sd t2, 48(sp)  # x7
    sd s0, 56(sp)  # x8
    sd s1, 64(sp)  # x9
    sd a0, 72(sp)  # x10
    sd a1, 80(sp)  # x11
    sd a2, 88(sp)  # x12
    sd a3, 96(sp)  # x13
    sd a4, 104(sp) # x14
    sd a5, 112(sp) # x15
    sd a6, 120(sp) # x16
    sd a7, 128(sp) # x17
    sd s2, 136(sp) # x18
    sd s3, 144(sp) # x19
    sd s4, 152(sp) # x20
    sd s5, 160(sp) # x21
    sd s6, 168(sp) # x22
    sd s7, 176(sp) # x23
    sd s8, 184(sp) # x24
    sd s9, 192(sp) # x25
    sd s10, 200(sp)# x26
    sd s11, 208(sp)# x27
    sd t3, 216(sp) # x28
    sd t4, 224(sp) # x29
    sd t5, 232(sp) # x30
    sd t6, 240(sp) # x31

    # 2. Call the C interrupt handler
    call timer_handler

    # 3. Restore the full context from the stack

    ld t6, 240(sp) # x31
    ld t5, 232(sp) # x30
    ld t4, 224(sp) # x29
    ld t3, 216(sp) # x28
    ld s11, 208(sp)# x27
    ld s10, 200(sp)# x26
    ld s9, 192(sp) # x25
    ld s8, 184(sp) # x24
    ld s7, 176(sp) # x23
    ld s6, 168(sp) # x22
    ld s5, 160(sp) # x21
    ld s4, 152(sp) # x20
    ld s3, 144(sp) # x19
    ld s2, 136(sp) # x18
    ld a7, 128(sp) # x17
    ld a6, 120(sp) # x16
    ld a5, 112(sp) # x15
    ld a4, 104(sp) # x14
    ld a3, 96(sp)  # x13
    ld a2, 88(sp)  # x12
    ld a1, 80(sp)  # x11
    ld a0, 72(sp)  # x10
    ld s1, 64(sp)  # x9
    ld s0, 56(sp)  # x8
    ld t2, 48(sp)  # x7
    ld t1, 40(sp)  # x6
    ld t0, 32(sp)  # x5
    ld tp, 24(sp)  # x4
    ld gp, 16(sp)  # x3
    ld ra, 0(sp)   # x1

    # Deallocate stack frame
    addi sp, sp, 256

    # 4. Return from interrupt
    mret