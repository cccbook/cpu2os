# FreeRTOS 在 RISC-V 上執行

* https://github.com/ccc-c/FreeRTOS-Kernel

## 官方專案

* https://github.com/FreeRTOS
    * https://github.com/FreeRTOS/FreeRTOS-Kernel/tree/master/portable/GCC/RISC-V
* https://www.freertos.org/RTOS-RISC-V-FreedomStudio-QMEU.html
    * https://interactive.freertos.org/hc/en-us/community/posts/210030246-32-bit-and-64-bit-RISC-V-using-GCC

## 以下是非官方專案

Clone 下列專案：

* https://github.com/illustris/FreeRTOS-RISCV

然後在 misavo.com 上用下列方式 make

```
root@localhost:/home/guest/FreeRTOS-RISCV/Demo/riscv-spike# make RISCV=/usr/local
... // 之後有些警告，沒關係
    CC ../Common/Minimal/blocktim.c
    CC ../Common/Minimal/countsem.c
    CC ../Common/Minimal/recmutex.c
Linking....
Completed riscv-spike.elf
```

然後就可以用 spike 執行，記憶體預設 2G 太大，調成 256MB 就好了。

```
root@localhost:/home/guest/FreeRTOS-RISCV/Demo/riscv-spike# spike -m256 riscv-spike.elf
```

## 參考文獻

* https://www.freertos.org/RTOS-RISC-V-FreedomStudio-QMEU.html (這份應該是我要的！)
* https://www.freertos.org/Using-FreeRTOS-on-RISC-V.html
* http://wiki.csie.ncku.edu.tw/embedded/freertos
    * jserv 和學生寫的 FreeRTOS 技術報告
