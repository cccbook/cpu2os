# 專案建置工具 Make

當程式越來越多時，編譯、連結與測試的動作會變得相當繁瑣，此時就必須使用專案建置工具。GNU 的 make 是專案編譯上相當著名的典型工具，在此，我們將用 make 來學習大型專案開發所需的專案管理技巧。並且透過 make 觀察大型專案的開發過程，讓讀者得以學習到專業的系統程式開發流程。

專案編譯工具 make 是用來編譯專案的強有力工具，使用 make 工具可以有效的整合許多程式，並且進行快速的大型專案編譯動作。像是著名的 Linux 作業系統就是以 gcc 與 make 等 GNU 工具所建構出來的。因此，學習 GNU 工具更是進入 Linux 系統程式設計的捷徑。

對於初學者而言，可能會覺得 make 的語法相當怪異，然而，對於有經驗的程式設計人員而言，卻會覺得 make 專案管理工具相當方便。

## Makefile 特殊符號

```
$@ : 該規則的目標文件 (Target file)
$* : 代表 targets 所指定的檔案，但不包含副檔名
$< : 依賴文件列表中的第一個依賴文件 (Dependencies file)
$^ : 依賴文件列表中的所有依賴文件
$? : 依賴文件列表中新於目標文件的文件列表

?= 語法 : 若變數未定義，則替它指定新的值。
:= 語法 : make 會將整個 Makefile 展開後，再決定變數的值。
```

## 範例 1 ：

```
%.o: %.c
    gcc -c $< -o $@

$< : 屬於第一條件，也就是 foo.c
$@ : 屬於目標條件，也就是 foo.o
```

## 範例 2 : https://github.com/jserv/mini-arm-os/blob/master/07-Threads/Makefile



## 參考

* [EricWang: makefile 心得、教學](https://wwssllabcd.github.io/blog/2016/10/03/how-to-write-make-file/)
* [Makefile 語法和示範](https://hackmd.io/@sysprog/SySTMXPvl)

