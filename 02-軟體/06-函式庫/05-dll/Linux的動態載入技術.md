# Linux 的動態載入技術

在 Linux 的 2.6 版當中支援了動態連結技術，因而也支援了動態載入函式庫。
Linux 的動態載入函式庫儲存在 libdl.so 這個檔案中，您可以透過引用 `<dlfcn.h>` 這個檔案，使用 Linux 的動態載入函式庫。

以下範例 顯示了 Linux 動態載入的程式範例，該程式使用 dlopen 載入數學函式庫 libm.so，然後用 dlsym 取得 cos() 函數的指標，接著呼叫該函數印出 cos(2.0) 的值，最後用 dlclose() 關閉函式庫。

File: dlcall.c

```cpp
#include <dlfcn.h>

int main(void) {
  void *handle = dlopen("libm.so", RTLD_LAZY);
  double (*cosine)(double);
  cosine = dlsym(handle, "cos");
  printf ("%f\n", (*cosine)(2.0));
  dlclose(handle);
  return 0;
}
```

編譯執行

```
$ gcc -o dl_call dl_call.c –ldl

```

這種動態載入功能對系統程式設計師而言非常有用，舉例而言，為了降低所使用的記憶體數量，您可以利用動態載入功能，在需要時才載入某個函式庫，並且在用完後立即釋放。如此，就能在程式尚未結束前，就將函式庫所佔據的空間還給
作業系統，因而降低的記憶體的用量。或者，您也可以在使用者輸入某些參數，像是函式庫名稱之後，才用程式動態的載入該函數庫，如此，就不需要事先將所有可能用到的函式庫全部引用進來，因而解決了記憶體不足的現象。

