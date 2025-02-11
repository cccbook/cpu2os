# 系統程式 -- 習題

* 示範程式 -- https://github.com/ccccourse/sp

## 選項一 -- 從 C 到組合語言

1. 請將以下編譯器擴充，讓他可以處理 IF 語法
    * https://github.com/cccbook/sp/tree/master/code/c/02-compiler/03-compiler
2. 請寫出第一題的中間碼之虛擬機
    * 可以用解譯的方式做就行 (速度較慢，但測試比較方便，問題會比較少!)
    * 也可以用 JIT 立即編譯的方式做 (速度較快，但測試比較難，問題會比較多!)
    * 參考 -- https://github.com/cccbook/sp/tree/master/code/c/03-asm/04-jitCall/win
3. 請寫出將中間碼轉為組合語言的程式
    * 可以轉為 x86, RISC-V, 或 nand2tetris 的 hackCPU 組合語言
    * 參考 (c0c) -- https://github.com/ccc-c/c0c
4. 請寫出第 3 題組合語言的組譯器
    * 若選 x86 的話，可能要自訂指令集
    * 若選 RISC-V 的話，要弄清楚指令編碼規則
    * 若選 hackCPU 是最簡單的 (老師也有寫範例)
    * 參考 (c0c) -- https://github.com/ccc-c/c0c
5. 請寫出第四題輸出機器碼可執行的虛擬機
    * 參考 (C 語言) -- https://github.com/cccbook/sp/tree/master/code/c/04-machine/nand2tetris/vm

## 選項二 -- nand2tetris 

這組選項老師都有寫範例 (1,2 有 js+c 版本， 3, 4 只有 js 版本)

1. 請寫出 nand2tetris 的 HackCPU 之虛擬機
    * 參考 (C 語言) -- https://github.com/cccbook/sp/tree/master/code/c/04-machine/nand2tetris/vm
2. 請寫出寫 nand2tetris HackCPU 的組譯器
    * 參考 (C 語言)-- https://github.com/cccbook/sp/tree/master/code/c/04-machine/nand2tetris/asm
3. 請寫出寫 nand2tetris 第 7, 8 兩章的虛擬機
    * 參考 (JavaScript) -- https://github.com/cccbook/sp/tree/master/code/nand2tetris/08/js
4. 請寫出寫 nand2tetris 第 9, 10, 11 三章的編譯器
    * 參考 -- https://github.com/cccbook/sp/tree/master/code/nand2tetris/11/js
5. 請閱讀 nand2tetris 第 12 章的作業系統，並對程式碼加上你的註解。
    * 參考 -- https://github.com/havivha/Nand2Tetris/tree/master/12


