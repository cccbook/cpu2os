# Context switch (內文切換，C 語言呼叫方法如下)
#
#   void swtch(struct context *old, struct context *new);
# 
# Save current registers in old. Load from new.	
# 問題：為何不用儲存全部暫存器？ 
# 猜測：這應該看編譯器會用到那些暫存器當臨時變數，像是參數 a0..a7 應該就不用保存。
.globl swtch
swtch: # 內文切換，保存舊暫存器群，切換到新暫存器群
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
        
        ret

	
