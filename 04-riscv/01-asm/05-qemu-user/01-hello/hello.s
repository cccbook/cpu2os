        .section .data
message: .asciz "Hello, world\n"  # 要輸出的字串

        .section .text
        .globl main
main:
        # 寫入 stdout (檔案描述符 1)
        li a7, 64               # 將系統呼叫號 64 (write) 載入到 a7
        li a0, 1                # 檔案描述符 1 是 stdout
        la a1, message          # 將字串的地址載入 a1
        li a2, 13               # 字串長度（"Hello, world\n" 的長度為 13）
        ecall                   # 呼叫系統以執行寫入

        # 結束程式
        li a7, 93               # 將系統呼叫號 93 (exit) 載入到 a7
        xor a0, a0, a0         # 返回值為 0
        ecall                   # 呼叫系統以結束程式
