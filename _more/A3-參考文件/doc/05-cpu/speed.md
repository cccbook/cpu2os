# 處理器效能 -- 速度議題

# 簡介

* [現代處理器設計](http://hackfoldr.org/cpu/) -- jserv

```
CPU 時間 = CPU 執行一個指令所需的時脈週期 / 時脈頻率

CPI = CPU 執行一個指令所需的時脈週期 / 指令數

CPU 時間 = 指令數 * 每個指令週期數 * 時脈週期時間
```

# 安達荷定理 (Amdahl's Law)

改善一系統的某部分以提升效能之時，其改善程度會受限於「該部分所佔的時間比率」。

```
    S = 1/(1-f)+f/k

S : 整體系統的改善比率。
f: 改善部分的比率。
k: 該部分系統改善後的效能增加倍數。
```

## RISC 精簡指令集電腦

使用管線 (Pipeline) 讓電腦加速

* 管線 CPI = 理想管線 CPI + 結構暫停值 + 資料違障暫停值 + 流程控制暫停值

## 動態排程 

Tomasulo 演算法

## 利用快取增加速度

* 目錄性快取一致性協定

## 記憶體階層 (Memory Hierarchy)

* 記憶裝置
 * SRAM
 * DRAM
 * 磁碟機

* 分頁式虛擬記憶體
 * TLB : Translate Look aside Buffer

* 輸出入系統
 * RAID : 磁碟陣列

## 參考文獻

* [陳鍾誠:如何設計電腦 -- 還有讓電腦變快的那些方法](https://www.slideshare.net/ccckmit/ss-85466673) (SlideShare)
* [Jserv : 現代處理器設計：原理和關鍵特徵](http://hackfoldr.org/cpu/)
* [C 語言pthread 多執行緒平行化程式設計入門教學與範例- G. T. Wang](https://blog.gtwang.org/programming/pthread-multithreading-programming-in-c-tutorial/)
* [維基百科:超執行緒](https://zh.wikipedia.org/wiki/%E8%B6%85%E5%9F%B7%E8%A1%8C%E7%B7%92)
* [CPU Cache 原理探討](https://hackmd.io/s/H1U6NgK3Z)
* [維基百科:平行計算](https://zh.wikipedia.org/wiki/%E5%B9%B6%E8%A1%8C%E8%AE%A1%E7%AE%97)
* [維基百科:費林分類法](https://zh.wikipedia.org/wiki/%E8%B2%BB%E6%9E%97%E5%88%86%E9%A1%9E%E6%B3%95)
* [維基百科:指令管線化](https://zh.wikipedia.org/wiki/%E6%8C%87%E4%BB%A4%E7%AE%A1%E7%B7%9A%E5%8C%96)
* [維基百科:圖形處理器](https://zh.wikipedia.org/wiki/%E5%9C%96%E5%BD%A2%E8%99%95%E7%90%86%E5%99%A8)
* [什麼是 GPU 加速運算？](http://www.nvidia.com.tw/object/what-is-gpu-computing-tw.html)
* [Google 硬體工程師揭密，TPU 為何會比 CPU、GPU 快 30 倍](https://technews.tw/2017/04/07/first-in-depth-look-at-googles-tpu-architecture/)
* [免費電子書：Verilog 電路設計](http://ccckmit.wikidot.com/ve:main)
* 開源指令集架構（ISA） -- https://en.wikipedia.org/wiki/RISC-V
