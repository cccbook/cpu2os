/* trap.S */
.section .text.boot
.globl _start
_start:
    # 1. 初始化主堆疊 (OS 核心堆疊)
    #    (注意: linker script 中的 _stack_top 是高位址)
    la sp, _stack_top

    # 2. 設置 M-Mode 陷阱向量 (Trap Vector)
    #    告訴 CPU 發生中斷/異常時跳轉到 m_mode_trap_handler
    la t0, m_mode_trap_handler
    csrw mtvec, t0

    # 3. 跳轉到 C 語言的 OS 主函式
    call os_main

    # 4. 如果 os_main 返回 (不應該發生), 進入無限迴圈
    j .

.globl m_mode_trap_handler
m_mode_trap_handler:
    # -----------------------------------------------
    # 1. 交換 sp 與 mscratch
    #    執行後: sp = 核心堆疊/上下文指標
    #             mscratch = 任務的使用者堆疊指標
    # -----------------------------------------------
    csrrw sp, mscratch, sp

    # -----------------------------------------------
    # 2. CONTEXT_SAVE
    #    分配上下文空間 (33 words = 132 bytes)
    # -----------------------------------------------
    addi sp, sp, -132

    # 儲存 GPRs (x1, x3-x31)
    sw x1, 0(sp)     # ra
    # x2 (sp) 比較特別, 我們要存的是 mscratch 裡的值
    csrr t0, mscratch
    sw t0, 4(sp)     # user_sp
    
    sw x3, 8(sp)     # gp
    sw x4, 12(sp)    # tp
    sw x5, 16(sp)    # t0
    sw x6, 20(sp)    # t1
    sw x7, 24(sp)    # t2
    sw x8, 28(sp)    # s0
    sw x9, 32(sp)    # s1
    sw x10, 36(sp)   # a0
    sw x11, 40(sp)   # a1
    sw x12, 44(sp)   # a2
    sw x13, 48(sp)   # a3
    sw x14, 52(sp)   # a4
    sw x15, 56(sp)   # a5
    sw x16, 60(sp)   # a6
    sw x17, 64(sp)   # a7
    sw x18, 68(sp)   # s2
    sw x19, 72(sp)   # s3
    sw x20, 76(sp)   # s4
    sw x21, 80(sp)   # s5
    sw x22, 84(sp)   # s6
    sw x23, 88(sp)   # s7
    sw x24, 92(sp)   # s8
    sw x25, 96(sp)   # s9
    sw x26, 100(sp)  # s10
    sw x27, 104(sp)  # s11
    sw x28, 108(sp)  # t3
    sw x29, 112(sp)  # t4
    sw x30, 116(sp)  # t5
    sw x31, 120(sp)  # t6

    # 儲存 CSRs
    csrr t0, mepc
    sw t0, 124(sp)
    csrr t0, mstatus
    sw t0, 128(sp)

    # -----------------------------------------------
    # 3. 儲存目前任務的上下文指標 (sp)
    # -----------------------------------------------
    la t0, g_current_task_ctx_ptr
    sw sp, 0(t0)

    # -----------------------------------------------
    # 4. 處理陷阱
    # -----------------------------------------------
    csrr t0, mcause

    # 檢查是否為 M-Mode 計時器中斷 (mcause = 0x80000007)
    li t1, 0x80000007
    bne t0, t1, handle_exception # 如果不是, 跳轉到異常處理

handle_timer_irq:
    # 是計時器中斷, 呼叫 C 函式處理
    call timer_tick # 重新設定計時器
    call scheduler  # 呼叫排程器
    j restore_context

handle_exception:
    # 簡易異常處理 (無限迴圈卡住)
    j .

    # -----------------------------------------------
    # 5. CONTEXT_RESTORE
    # -----------------------------------------------
restore_context:
    # 載入 *下一個* 任務的上下文指標到 sp
    la t0, g_next_task_ctx_ptr
    lw sp, 0(t0)

restore_context_no_load:
    # 恢復 CSRs
    lw t0, 124(sp)
    csrw mepc, t0
    lw t0, 128(sp)
    csrw mstatus, t0

    # 恢復 GPRs (x1, x3-x31)
    lw x1, 0(sp)
    # (跳過 x2)
    lw x3, 8(sp)
    lw x4, 12(sp)
    lw x5, 16(sp)
    lw x6, 20(sp)
    lw x7, 24(sp)
    lw x8, 28(sp)
    lw x9, 32(sp)
    lw x10, 36(sp)
    lw x11, 40(sp)
    lw x12, 44(sp)
    lw x13, 48(sp)
    lw x14, 52(sp)
    lw x15, 56(sp)
    lw x16, 60(sp)
    lw x17, 64(sp)
    lw x18, 68(sp)
    lw x19, 72(sp)
    lw x20, 76(sp)
    lw x21, 80(sp)
    lw x22, 84(sp)
    lw x23, 88(sp)
    lw x24, 92(sp)
    lw x25, 96(sp)
    lw x26, 100(sp)
    lw x27, 104(sp)
    lw x28, 108(sp)
    lw x29, 112(sp)
    lw x30, 116(sp)
    lw x31, 120(sp)
    
    # -----------------------------------------------
    # 6. 恢復 x2 (sp) 和 mscratch
    # -----------------------------------------------
    # 恢復 mscratch (裡面是新任務的 *核心* 堆疊指標, 即 sp)
    csrw mscratch, sp 
    # 恢復任務的 *使用者* 堆疊指標到 x2 (sp)
    lw x2, 4(sp)
    
    # -----------------------------------------------
    # 7. 從陷阱返回
    # -----------------------------------------------
    mret

.globl launch_first_task
launch_first_task:
    # 這是 os_main() 最後呼叫的函式
    # 它只執行一次, 用來啟動第一個任務
    # 它不需要儲存, 只需要 *恢復*
    j restore_context
