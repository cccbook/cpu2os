# gcc

* [維基百科: GCC](https://zh.wikipedia.org/wiki/GCC)

## 簡介

gcc (GNU 編譯器工具集, 英語：GNU Compiler Collection），是最廣泛使用的 C 語言編譯器。

Linux 就是建構在 gcc 的基礎之上發展出來的，因此也稱為 GNU/Linux。

gcc 工具集不只包含 gcc 這個 C 語言編譯器，還包含 C++ 編譯器 g++，另外還支援了 go, java, ada, fortran 等語言。 

Windows 中的 Code Blocks, Dev C++, MinGW, MSYS2 等 C 語言開發平台，也都是以 gcc 為預設的編譯器。

GNU 是開放原始碼的創始組織，所提出的 GPL 授權強制使用該程式碼的人，也必須要開放原始碼。

## 安裝

1. Linux : 完全沒問題，最好的做法！
2. Mac : 安裝 brew 並安裝 gcc !
3. Windows :
    1. 直接使用 Code Blocks 或 Dev C++ 中的 gcc
    2. 安裝 MSYS2 並安裝 gcc

MSYS2 安裝 gcc

```
$ pacman -S glib2
$ pacman -S vim
$ pacman -S gcc
```
