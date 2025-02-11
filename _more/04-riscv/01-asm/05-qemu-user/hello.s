.section .data
msg:    .asciz "hello\n"  # 要輸出的字串，包含換行符

.section .text
.globl main
main:
    # 載入要輸出的字串的地址
    la a0, msg            # 將 msg 的地址載入 a0

    # 系統呼叫：打印字串
    li a7, 4              # 系統呼叫號，4 表示打印字串
    ecall                 # 呼叫系統

    # 結束程式
    li a7, 10             # 系統呼叫號，10 表示結束程式
    ecall                 # 呼叫系統
