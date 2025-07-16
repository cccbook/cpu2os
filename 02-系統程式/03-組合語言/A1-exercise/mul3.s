        .globl  mul3
        
        .text
mul3:
        mov     %rdi, %rax
        imulq   %rsi, %rax
        imulq   %rdx, %rax
        ret
