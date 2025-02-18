# GCC 簡介

## 什麼是 GCC？
GCC（GNU Compiler Collection）是一個強大且廣泛使用的開源編譯器集合，支援多種程式語言，包括 C、C++、Objective-C、Fortran、Ada、Go 及 D。它由 GNU 計畫開發，最初於 1987 年由 Richard Stallman 釋出，作為自由軟體運動的一部分。

## 主要功能與特點
1. **跨平台支援**  
   GCC 可在各種作業系統上運行，包括 Linux、macOS、Windows（透過 MinGW 或 Cygwin）等，並支援多種硬體架構。

2. **多語言支援**  
   除了 C 和 C++ 之外，GCC 也支援多種其他程式語言，使其成為一個通用的編譯工具。

3. **最佳化（Optimization）**  
   GCC 提供了多種最佳化選項，如 `-O1`、`-O2`、`-O3` 和 `-Os`，可提高執行效率或減少程式大小。

4. **可擴充性**  
   透過 GCC 插件機制，開發者可以擴充其功能，例如新增警告、錯誤檢查或特定最佳化技術。

5. **開源自由**  
   GCC 採用 GPL（GNU General Public License）授權，允許開發者自由使用、修改和發佈。

## GCC 的基本使用
GCC 的基本使用方式如下：

### 1. 編譯 C 程式
假設有一個名為 `hello.c` 的 C 程式：
```c
#include <stdio.h>

int main() {
    printf("Hello, GCC!\n");
    return 0;
}
```
可使用以下指令進行編譯：
```sh
gcc hello.c -o hello
```
執行產生的執行檔：
```sh
./hello
```

### 2. 編譯 C++ 程式
對於 C++ 程式，可使用 `g++`：
```sh
g++ hello.cpp -o hello
```

### 3. 使用最佳化
可加入 `-O2` 進行最佳化：
```sh
gcc -O2 hello.c -o hello
```

### 4. 啟用警告
使用 `-Wall` 顯示所有常見的警告：
```sh
gcc -Wall hello.c -o hello
```

### 5. 產生偵錯資訊
使用 `-g` 選項可讓程式更容易進行偵錯：
```sh
gcc -g hello.c -o hello
```
然後可使用 `gdb` 進行偵錯：
```sh
gdb ./hello
```

## GCC 的應用領域
- **系統開發**：Linux 核心及許多系統工具都是使用 GCC 編譯的。
- **嵌入式系統**：GCC 支援 ARM、RISC-V 等多種架構，是嵌入式開發的重要工具。
- **科學運算**：透過 GCC 支援的 Fortran 和 C，可以進行高效能計算（HPC）。
- **學術研究**：許多學術專案與自由軟體專案使用 GCC 作為主要編譯工具。

## 結論
GCC 是一個功能強大、靈活且高效的編譯器，廣泛應用於軟體開發、系統開發及學術研究。作為開源工具，它持續發展，並在現代軟體開發中發揮關鍵作用。無論是初學者還是專業開發者，熟練掌握 GCC 都能提升程式開發的效率與品質。