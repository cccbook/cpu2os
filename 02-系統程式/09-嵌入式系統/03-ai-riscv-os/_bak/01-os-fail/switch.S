.section .text
.globl timer_vector
.globl switch_to

# ============================================================================
# timer_vector
# Machine-mode timer interrupt handler.
# This function is responsible for:
# 1. Saving the context of the currently running thread.
# 2. Calling the C-level handler (timer_handler) which calls the scheduler.
# 3. Restoring the context of the next thread to run.
# ============================================================================
.align 4
timer_vector:
    # --- 1. Save the context of the current thread ---
    
    # Load the address of the global 'threads' array and 'current_thread' index
    .extern threads
    .extern current_thread

    # Calculate the address of the current TCB's context
    # t0 = &current_thread
    # t1 = value of current_thread
    # t2 = &threads
    # t3 = sizeof(struct tcb)
    # t4 = address of current tcb (&threads[current_thread])
    # t5 = address of current context (&threads[current_thread].context)
    la   t0, current_thread
    lw   t1, 0(t0)        
    la   t2, threads
    
    # Calculate sizeof(struct tcb)
    # struct context: 32 registers * 8 bytes/reg = 256 bytes
    # uint64_t stack[1024]: 1024 * 8 bytes = 8192 bytes
    # int state: 4 bytes, padded to 8 bytes for alignment. Total size is 8.
    # Total = 256 + 8192 + 8 = 8456 bytes
    li   t3, 8456
    
    mul  t4, t1, t3
    add  t5, t2, t4  # t5 now holds the base address of the context

    # Save all general-purpose registers to the context structure
    sd ra, 0(t5)
    sd sp, 8(t5)
    sd gp, 16(t5)
    sd tp, 24(t5)
    sd t0, 32(t5)
    sd t1, 40(t5)
    sd t2, 48(t5)
    sd s0, 56(t5)
    sd s1, 64(t5)
    sd a0, 72(t5)
    sd a1, 80(t5)
    sd a2, 88(t5)
    sd a3, 96(t5)
    sd a4, 104(t5)
    sd a5, 112(t5)
    sd a6, 120(t5)
    sd a7, 128(t5)
    sd s2, 136(t5)
    sd s3, 144(t5)
    sd s4, 152(t5)
    sd s5, 160(t5)
    sd s6, 168(t5)
    sd s7, 176(t5)
    sd s8, 184(t5)
    sd s9, 192(t5)
    sd s10, 200(t5)
    sd s11, 208(t5)
    sd t3, 216(t5)
    sd t4, 224(t5)
    sd t5, 232(t5)
    sd t6, 240(t5)

    # Save mepc (Machine Exception Program Counter)
    csrr t0, mepc
    sd   t0, 256(t5) # Offset of mepc in struct context is 32*8=256

    # --- 2. Call the C handler ---
    call timer_handler

    # --- 3. Restore the context of the next thread ---
    # The 'current_thread' index may have been changed by the scheduler.
    # We must recalculate the context address.
    la   t0, current_thread
    lw   t1, 0(t0)
    la   t2, threads
    li   t3, 8456 # sizeof(struct tcb)
    mul  t4, t1, t3
    add  t5, t2, t4 # t5 now holds the base address of the NEW context

    # Restore mepc
    ld   t0, 256(t5)
    csrw mepc, t0
    
    # Restore all general-purpose registers
    ld ra, 0(t5)
    # We load sp last
    ld gp, 16(t5)
    ld tp, 24(t5)
    ld t0, 32(t5)
    ld t1, 40(t5)
    ld t2, 48(t5)
    ld s0, 56(t5)
    ld s1, 64(t5)
    ld a0, 72(t5)
    ld a1, 80(t5)
    ld a2, 88(t5)
    ld a3, 96(t5)
    ld a4, 104(t5)
    ld a5, 112(t5)
    ld a6, 120(t5)
    ld a7, 128(t5)
    ld s2, 136(t5)
    ld s3, 144(t5)
    ld s4, 152(t5)
    ld s5, 160(t5)
    ld s6, 168(t5)
    ld s7, 176(t5)
    ld s8, 184(t5)
    ld s9, 192(t5)
    ld s10, 200(t5)
    ld s11, 208(t5)
    ld t3, 216(t5)
    ld t4, 224(t5)
    ld t5, 232(t5)
    ld t6, 240(t5)
    
    # Restore sp last
    ld sp, 8(t5)

    # --- 4. Return from interrupt ---
    mret

# ============================================================================
# switch_to(struct context *new_context)
# Starts the first thread by loading its context and executing mret.
# The argument (pointer to the context) is passed in register a0.
# ============================================================================
switch_to:
    # a0 holds the address of the new context
    
    # Restore mepc
    ld   t0, 256(a0)
    csrw mepc, t0

    # Restore all general-purpose registers from the new context
    ld ra, 0(a0)
    # We load sp last
    ld gp, 16(a0)
    ld tp, 24(a0)
    ld t0, 32(a0)
    ld t1, 40(a0)
    ld t2, 48(a0)
    ld s0, 56(a0)
    ld s1, 64(a0)
    # Skip a0 for now, as it's our base pointer
    ld a1, 80(a0)
    ld a2, 88(a0)
    ld a3, 96(a0)
    ld a4, 104(a0)
    ld a5, 112(a0)
    ld a6, 120(a0)
    ld a7, 128(a0)
    ld s2, 136(a0)
    ld s3, 144(a0)
    ld s4, 152(a0)
    ld s5, 160(a0)
    ld s6, 168(a0)
    ld s7, 176(a0)
    ld s8, 184(a0)
    ld s9, 192(a0)
    ld s10, 200(a0)
    ld s11, 208(a0)
    ld t3, 216(a0)
    ld t4, 224(a0)
    ld t5, 232(a0)
    ld t6, 240(a0)
    
    # Restore a0
    ld a0, 72(a0)

    # Restore sp last
    ld sp, 8(a0)

    # "Return" to the new thread's entry point
    mret
