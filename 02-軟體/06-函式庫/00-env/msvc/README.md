## MsVC

微軟的 c 語言編譯器稱為 cl ，

https://docs.microsoft.com/zh-tw/cpp/build/reference/compiler-options

安裝好 Visual Studio 之後，請開啟 Developer PowerShell for VS2019

然後切到程式所在位置，輸入下列指令，然後 add.dll 就會被產生出來

```
PS C:\ccc\course\ws\14-more\callc> cl /LD add.c /link /EXPORT:add
Microsoft (R) C/C++ Optimizing Compiler Version 19.29.30133 for x86
Copyright (C) Microsoft Corporation.  著作權所有，並保留一切權利。

add.c
Microsoft (R) Incremental Linker Version 14.29.30133.0
Copyright (C) Microsoft Corporation.  All rights reserved.

/out:add.dll
/dll
/implib:add.lib
/EXPORT:add
add.obj
   正在建立程式庫 add.lib 和物件 add.exp
```

然後執行 ffi.ts 時發生錯誤

```
PS C:\ccc\course\ws\14-more\callc> deno run --allow-ffi --unstable ffi.ts
error: Uncaught Error: Could not open library: %1 不是有效的 Win32 應用程式。 (os error 193)
const dylib = Deno.dlopen(libName, {
                   ^
    at deno:core/01_core.js:106:46
    at unwrapOpResult (deno:core/01_core.js:126:13)
    at Object.opSync (deno:core/01_core.js:140:12)
    at new DynamicLibrary (deno:ext/ffi/00_ffi.js:12:24)
    at Object.dlopen (deno:ext/ffi/00_ffi.js:26:12)
    at file:///C:/ccc/course/ws/14-more/callc/ffi.ts:20:20
```

## 參考

微軟 c 語言編譯器 cl 的路徑 

C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64