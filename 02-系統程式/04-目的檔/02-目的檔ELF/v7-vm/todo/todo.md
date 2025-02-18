## 未解決問題

像 puts("hello") 這樣的字串，定址會有問題，如何解決？

=> 將 data 定在某個位址開始，像 stack 一樣，然後用特定暫存器指向該位址。


## 已解決問題

main, sum 這樣的函數指向不同位址?? NO


vm_test 時

call sum // sum 位址為 0

結果得到

```
0070 auipc x1, 0        # x[1]=0x70
0074 jalr  x1, 0(x1)    # x[1]=pc+4, goto 0x78
0078 sw    x10, -20(x8) # m[7980]=x10=10 
```

問題是按照下列實作，這樣不會跳轉到 0 ，而是根本不會跳。（問題在哪？）

```c
void auipc(VM* vm, uint8_t rd, uint32_t imm) {
    // 獲取當前 PC 的值
    uint32_t current_pc = vm->pc;

    // 計算高 20 位立即數的值
    uint32_t upper_imm = (imm << 12) & 0xFFFFF000; // 擴展至 32 位

    // 將結果存儲到指定寄存器
    vm->registers[rd] = current_pc + upper_imm;
}
```

解決了，因為 gcc 設定 -c 只編譯不連結，所以 objdump 反組譯就錯了。

拿掉 -c 改只用 -nostdlib ，然後加上  -Wl,--section-start=.text=0x0 從 0 開始，就可以用了。


