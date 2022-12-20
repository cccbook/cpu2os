# gcc 開發工具

gcc 是 GNU C Compiler 的簡稱，是一組非常重要的開放原始碼編譯器工具鏈，除了編譯器之外，也包含了《組譯器 as、連結器 ld、目的檔工具 bintool》 等。


## 使用

在此讓我們介紹 GNU 工具的使用方法，我們將利用範例導向的方式，導引讀者熟悉GNU 工具的操作。圖  1 .4 顯示了GNU工具使用的基本流程。

![](./img/gccFlow.png)

圖 1.4 GNU工具使用的基本流程

在使用 GNU 工具時，圖  1 .4 中的 printf 等函數，通常會是事先被函式庫工具 ar 建立好放入 libc.a 這個檔案中。因此使用者不需再重新建立這些函式庫。程式人員只要負責撰寫 main.c、sum.c 等應用程式，然後利用 gcc 編譯器進行編譯，再用 ld 連結器連結後，輸出執行檔即可。

根據這樣的流程，現在，就請讀者撰寫如範例 1 .2的兩個程式 main.c 與 sum.c。

範例 : main.c  

```c
#include <stdio.h>

int main(void) {
  int sum1 = sum(10);
  printf("sum=%d\n", sum1);
  system("pause");
  return 1;
}
```

範例 : sum.c  

```c
int sum(int n) {
  int s=0;
  int i;
  for (i=1; i<=n;i++) {
     s = s + i;
  }
  return s;
}
```

接著，您可以使用 GNU 的 gcc 工具對這些檔案進行編譯連結，直接產生執行檔，以下是其執行結果。

```
$ gcc sum.c main.c -o sum

$ gcc sum.c main.c -o sum

$ dir *.exe
 磁碟區 C 中的磁碟沒有標籤。
 磁碟區序號:  70AE-6E8A

 C:\ch01 的目錄

2010/03/12  上午 09:01            16,019 sum.exe
               1 個檔案          16,019 位元組
               0 個目錄  11,429,384,192 位元組可用

$ sum
sum=55
```

圖 : 利用 gcc同時編譯 main.c 與 sum.c 並輸出執行檔

在上圖中，gcc sum.c main.c -o sum 指令所指定的執行檔名稱為 sum，在 Windows 的環境中，由於預設的輸出副檔名是 .exe，因此您會看到一個 sum.exe 的執行檔被產生出來。

接著，我們將示範如何用 gcc 將 C 語言編譯後轉換為組合語言。這可以利用 gcc 中大寫的 -S 參數完成，參數 -S 用來告訴 gcc應該產生組合語言而非執行檔。如此，有利於我們觀察組合語言的寫法，以下兩個指令可以分別將程式 sum.c 與 main.c轉換為組合語言。

```
$ gcc -S sum.c -o sum.s
$ gcc -S main.c -o main.s
```

gcc 雖然是個編譯器，但是也可以作為組譯器使用。因此，我們可以利用gcc 編譯器來『組譯』組合語言。在圖  1 .6中，我們示範了如何利用 gcc 當作組譯器，將 sum.s 與main.s兩個組合語言程式，組譯後立即連結為執行檔 sum2。

```
$ gcc -S sum.c -o sum.s

$ gcc -S main.c -o main.s

$ gcc main.s sum.s -o sum2

$ sum2
sum=55
```

圖 : 將 gcc 當成組譯器使用

除了當組譯器使用之外，gcc 更能將 C 語言與組合語言檔案混合輸入，以單一指令完成編譯、組譯、連結等動作。在圖  1 .7中，我們將示範如何利用 gcc，同時編譯 C 語言檔 main.c 與組合語言檔 sum.s，然後連結並產生執行檔 sum3。

```
$ gcc main.c sum.s -o sum3

$ sum3
sum=55
```

圖: 利用 gcc 編譯 C 語言 main.c 同時組譯組合語言 sum.s
