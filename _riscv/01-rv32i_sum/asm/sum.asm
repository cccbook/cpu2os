.text
.globl _start

_start:
    # --- 初始化 ---
    li t0, 0        # t0 用來儲存總和 (Sum)，初始為 0
    li t1, 1        # t1 用來當作計數器 (i)，初始為 1
    li t2, 10       # t2 設定為迴圈的上限 (10)

loop:
    # --- 檢查條件 ---
    # 如果 t1 (i) 大於 t2 (10)，則跳轉到 end_loop
    bgt t1, t2, end_loop 

    # --- 執行加總 ---
    add t0, t0, t1  # Sum = Sum + i (將 t1 加到 t0)

    # --- 更新計數器 ---
    addi t1, t1, 1  # i = i + 1 (將 t1 加 1)

    # --- 跳回迴圈開頭 ---
    j loop          # 無條件跳轉回 loop 標籤

end_loop:
    # --- 程式結束 ---
    # 此時 t0 暫存器內的值應為 55 (0x37)
    
    # 以下是用於模擬器 (如 RARS 或 Venus) 的結束系統呼叫
    # 將結果移動到 a0 (通常回傳值或參數放在 a0)
    mv a0, t0       
    
    # 呼叫系統結束 (Exit)
    li a7, 10       # 系統呼叫代號 10 代表 Exit
    ecall           # 執行系統呼叫