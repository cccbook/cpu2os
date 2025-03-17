# c4.c 主程式解析

這段程式碼是 c4.c 的主程式部分，負責整個編譯器的啟動、初始化以及執行。讓我詳細解說它的工作流程：

## compile() 函數

`compile()` 函數是實際的主程式邏輯，包含整個編譯流程的控制。

### 命令列參數處理

```c
dump = 0;
--argc; ++argv;
if (argc > 0 && **argv == '-' && (*argv)[1] == 's') { src = 1; --argc; ++argv; }
if (argc > 0 && **argv == '-' && (*argv)[1] == 'd') { debug = 1; --argc; ++argv; }
if (argc > 0 && **argv == '-' && (*argv)[1] == 'u') { dump = 1; --argc; ++argv; }
if (argc < 1) { printf("usage: c4 [-s] [-d] file ...\n"); return -1; }
```

這段程式碼處理命令列參數：
- `-s`：啟用源碼顯示模式，會輸出程式碼和生成的虛擬機指令
- `-d`：啟用調試模式，執行時會顯示每個指令
- `-u`：啟用符號表傾印模式，會顯示編譯後的符號表內容
- 最後必須提供一個源碼檔案名稱

### 記憶體配置

```c
poolsz = 256*1024; // arbitrary size
if (!(sym = malloc(poolsz))) { printf("could not malloc(%d) symbol area\n", poolsz); return -1; } // 符號段
if (!(le = e = malloc(poolsz))) { printf("could not malloc(%d) text area\n", poolsz); return -1; } // 程式段
if (!(data = malloc(poolsz))) { printf("could not malloc(%d) data area\n", poolsz); return -1; } // 資料段
if (!(sp = malloc(poolsz))) { printf("could not malloc(%d) stack area\n", poolsz); return -1; }  // 堆疊段
```

這部分為編譯器和虛擬機配置四個主要記憶體區域：
- `sym`：符號表，儲存變數和函數的相關資訊
- `e`（與 `le`）：程式/文本段，儲存生成的虛擬機指令
- `data`：資料段，儲存字符串常量等資料
- `sp`：堆疊段，用於函數呼叫和區域變數儲存

每個段的大小都預設為 256KB。

### 符號表初始化

```c
p = "char else enum if int return sizeof while "
    "open read close printf malloc free memset memcmp exit void main";
i = Char; while (i <= While) { next(); id[Tk] = i++; } // add keywords to symbol table
i = OPEN; while (i <= EXIT) { next(); id[Class] = Sys; id[Type] = INT; id[Val] = i++; } // add library to symbol table
next(); id[Tk] = Char; // handle void type
next(); idmain = id; // keep track of main
```

這段程式碼初始化符號表：
1. 添加關鍵字（`char`、`if`、`while` 等）
2. 添加系統函數（`open`、`read`、`printf` 等）
3. 處理 `void` 類型
4. 記錄 `main` 函數的標識符位置

使用 `next()` 函數將這些預定義標識符添加到符號表中。

### 讀取源碼

```c
if (!(lp = p = malloc(poolsz))) { printf("could not malloc(%d) source area\n", poolsz); return -1; }
if ((i = read(fd, p, poolsz-1)) <= 0) { printf("read() returned %d\n", i); return -1; }
p[i] = 0; // 設定程式 p 字串結束符號 \0
close(fd);
```

這部分從指定檔案讀取源碼：
1. 為源碼分配記憶體空間
2. 使用 `read()` 系統呼叫讀取檔案內容
3. 在源碼末尾添加 null 終止符
4. 關閉檔案

### 編譯與執行

```c
if (prog() == -1) return -1;

if (!(pc = (int *)idmain[Val])) { printf("main() not defined\n"); return -1; }
if (src) return 0;
if (dump) { sym_dump(sym); return 0; }

// setup stack
bp = sp = (int *)((int)sp + poolsz);
*--sp = EXIT; // call exit if main returns
*--sp = PSH; t = sp;
*--sp = argc;
*--sp = (int)argv;
*--sp = (int)t;
return run(pc, bp, sp);
```

這是主程式的核心部分：
1. 呼叫 `prog()` 函數進行編譯
2. 檢查是否有 `main()` 函數定義，若無則報錯
3. 如果是源碼顯示模式 (`src=1`)，則只編譯不執行
4. 如果是符號表傾印模式 (`dump=1`)，則顯示符號表後退出
5. 設定虛擬機的初始堆疊：
   - 設定 `bp` 和 `sp` 指向堆疊頂端
   - 壓入 `EXIT` 指令作為 `main()` 返回後的處理
   - 壓入初始參數：`argc` 和 `argv`
6. 呼叫 `run()` 函數開始執行程式

### 堆疊設定詳解

```c
bp = sp = (int *)((int)sp + poolsz);
*--sp = EXIT; // call exit if main returns
*--sp = PSH; t = sp;
*--sp = argc;
*--sp = (int)argv;
*--sp = (int)t;
```

這段程式碼設定初始堆疊框架：
1. 將 `bp` 和 `sp` 指向堆疊區域的高端
2. 壓入 `EXIT` 指令，當 `main()` 返回時自動呼叫
3. 壓入 `PSH` 指令，並記錄當前堆疊位置
4. 壓入 `argc`（命令列參數個數）
5. 壓入 `argv`（命令列參數陣列）
6. 壓入前面記錄的堆疊位置

這樣設定可確保 `main()` 函數能夠接收標準的命令列參數。

## main() 函數

```c
int main(int32_t argc, char **argv) { // 主程式
  compile(argc, argv);
}
```

實際的 `main()` 函數非常簡單，只是呼叫了 `compile()` 函數並傳遞命令列參數。

## 整體流程總結

整個程式的運作流程如下：
1. 處理命令列參數，設定運行模式
2. 分配各個記憶體區段（符號表、程式碼、資料、堆疊）
3. 初始化符號表，添加預定義標識符
4. 讀取源碼檔案內容
5. 呼叫 `prog()` 函數進行編譯
6. 根據運行模式決定後續動作：
   - 源碼顯示模式：僅編譯，不執行
   - 符號表傾印模式：顯示符號表，不執行
   - 正常模式：設定堆疊後執行程式
7. 執行時，從 `main()` 函數開始，將控制權交給虛擬機

這個結構非常簡潔，但實現了一個完整的編譯和執行環境，展示了最小化 C 語言實現所需的核心組件。