.section .data
result: .word 0          # 儲存結果的變數
n: .word 10              # 計算的上限 n

.section .text
.globl main              # 將 main 標記為全域
main:
    la a0, n            # 載入 n 的地址
    lw a1, 0(a0)        # 將 n 的值載入 a1
    li a2, 0            # 將 a2 設定為 0，用來累加

sum_loop:
    add a2, a2, a1      # 將 a1 加到 a2
    addi a1, a1, -1     # n = n - 1
    bgtz a1, sum_loop   # 如果 n > 0，繼續迴圈

    la a0, result       # 載入結果的地址
    sw a2, 0(a0)        # 將結果儲存到變數中

    # 結束程式
    li a7, 10           # 系統呼叫號，結束程式
    ecall
