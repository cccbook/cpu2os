# 如何在 Windows 中用 MinGW 建置 Lua ?

為了學習 Lua 的「編譯器與解譯器」的設計方式，筆者必須要自行建置 Lua 的環境，於是找到 Lua 官網的下載點如下：

* <http://www.lua.org/download.html>

筆者下載時為 5.2.3 版，但是從官網下載下列網址的壓縮檔後，發現檔案似乎已經損壞，因此無法解開。

* <http://www.lua.org/ftp/lua-5.2.3.tar.gz>

後來筆者乾脆到所有版本的下載點中：

* <http://www.lua.org/ftp/>

下載那個所有版本一次打包的檔案如下：

* <http://www.lua.org/ftp/lua-all.tar.gz>

結果就可以順利的解壓縮了！

於是我將解壓縮後的 lua-5.2.3 版之內容，放到 `c:\lua` 資料夾中。

接著我參考下列網址的說明：

* <http://stackoverflow.com/questions/16641826/how-do-i-build-lua-for-windows-using-mingw-and-msys>

在 MinGW 的環境中，使用 `make PLAT=mingw` 指令編譯 Lua 環境，結果得到下列錯誤訊息：

```
ccc@ccc-PC /c/lua
$ make PLAT=mingw
cc -Wall -O2   -c -o lapi.o lapi.c
make: cc: Command not found
make: *** [lapi.o] Error 127
```

看來是 make 的 cc 指令沒有定義，於是我在 Makefile 最前面一行加上 `CC=gcc` ，存檔後然後再下一次 `make PLAT=mingw` 指令，結果就可以了：

```
ccc@ccc-PC /c/lua
$ make PLAT=mingw
gcc -Wall -O2   -c -o lapi.o lapi.c
gcc -Wall -O2   -c -o lcode.o lcode.c
gcc -Wall -O2   -c -o lctype.o lctype.c
gcc -Wall -O2   -c -o ldebug.o ldebug.c
gcc -Wall -O2   -c -o ldo.o ldo.c
gcc -Wall -O2   -c -o ldump.o ldump.c
gcc -Wall -O2   -c -o lfunc.o lfunc.c
gcc -Wall -O2   -c -o lgc.o lgc.c
gcc -Wall -O2   -c -o llex.o llex.c
gcc -Wall -O2   -c -o lmem.o lmem.c
gcc -Wall -O2   -c -o lobject.o lobject.c
gcc -Wall -O2   -c -o lopcodes.o lopcodes.c
gcc -Wall -O2   -c -o lparser.o lparser.c
gcc -Wall -O2   -c -o lstate.o lstate.c
gcc -Wall -O2   -c -o lstring.o lstring.c
gcc -Wall -O2   -c -o ltable.o ltable.c
gcc -Wall -O2   -c -o ltm.o ltm.c
gcc -Wall -O2   -c -o lundump.o lundump.c
gcc -Wall -O2   -c -o lvm.o lvm.c
gcc -Wall -O2   -c -o lzio.o lzio.c
gcc -Wall -O2   -c -o lua.o lua.c
gcc -Wall -O2   -c -o lauxlib.o lauxlib.c
gcc -Wall -O2   -c -o lbaselib.o lbaselib.c
gcc -Wall -O2   -c -o lbitlib.o lbitlib.c
gcc -Wall -O2   -c -o lcorolib.o lcorolib.c
gcc -Wall -O2   -c -o ldblib.o ldblib.c
gcc -Wall -O2   -c -o liolib.o liolib.c
gcc -Wall -O2   -c -o lmathlib.o lmathlib.c
gcc -Wall -O2   -c -o loslib.o loslib.c
gcc -Wall -O2   -c -o lstrlib.o lstrlib.c
gcc -Wall -O2   -c -o ltablib.o ltablib.c
gcc -Wall -O2   -c -o loadlib.o loadlib.c
gcc -Wall -O2   -c -o linit.o linit.c
gcc -o lua lapi.o lcode.o lctype.o ldebug.o ldo.o ldump.o lfunc.o lgc.o llex.o l
mem.o lobject.o lopcodes.o lparser.o lstate.o lstring.o ltable.o ltm.o lundump.o
 lvm.o lzio.o lua.o lauxlib.o lbaselib.o lbitlib.o lcorolib.o ldblib.o liolib.o
lmathlib.o loslib.o lstrlib.o ltablib.o loadlib.o linit.o -lm
gcc -Wall -O2   -c -o luac.o luac.c
gcc -o luac lapi.o lcode.o lctype.o ldebug.o ldo.o ldump.o lfunc.o lgc.o llex.o
lmem.o lobject.o lopcodes.o lparser.o lstate.o lstring.o ltable.o ltm.o lundump.
o lvm.o lzio.o luac.o lauxlib.o -lm
./lua test.lua
Hello from Lua 5.2
./luac -l test.lua

main <test.lua:0,0> (6 instructions at 005CE140)
0+ params, 3 slots, 1 upvalue, 0 locals, 3 constants, 0 functions
        1       [1]     GETTABUP        0 0 -1  ; _ENV "print"
        2       [1]     LOADK           1 -2    ; "Hello from "
        3       [1]     GETTABUP        2 0 -3  ; _ENV "_VERSION"
        4       [1]     CONCAT          1 1 2
        5       [1]     CALL            0 2 1
        6       [1]     RETURN          0 1

```

看來是編譯成功了，編譯完成之後我們可以看到 ls 指令列出的檔案如下，裏面有 lua.exe 與 luac.exe 等兩個檔案，這分別是 Lua 的「解譯器」與 「編譯器」。

```
ccc@ccc-PC /c/lua
$ ls
Makefile    lcorolib.c  ldump.o   llimits.h   loslib.c   ltable.h   luac.o
lapi.c      lcorolib.o  lfunc.c   lmathlib.c  loslib.o   ltable.o   luac.out
lapi.h      lctype.c    lfunc.h   lmathlib.o  lparser.c  ltablib.c  luaconf.h
lapi.o      lctype.h    lfunc.o   lmem.c      lparser.h  ltablib.o  lualib.h
lauxlib.c   lctype.o    lgc.c     lmem.h      lparser.o  ltm.c      lundump.c
lauxlib.h   ldblib.c    lgc.h     lmem.o      lstate.c   ltm.h      lundump.h
lauxlib.o   ldblib.o    lgc.o     loadlib.c   lstate.h   ltm.o      lundump.o
lbaselib.c  ldebug.c    linit.c   loadlib.o   lstate.o   lua.c      lvm.c
lbaselib.o  ldebug.h    linit.o   lobject.c   lstring.c  lua.exe    lvm.h
lbitlib.c   ldebug.o    liolib.c  lobject.h   lstring.h  lua.h      lvm.o
lbitlib.o   ldo.c       liolib.o  lobject.o   lstring.o  lua.hpp    lzio.c
lcode.c     ldo.h       llex.c    lopcodes.c  lstrlib.c  lua.o      lzio.h
lcode.h     ldo.o       llex.h    lopcodes.h  lstrlib.o  luac.c     lzio.o
lcode.o     ldump.c     llex.o    lopcodes.o  ltable.c   luac.exe   test.lua
```

其中的 lua.c 是解譯器 lua.exe 的主程式，而 luac.c 則是編譯器 luac.exe 的主程式。

### 參考文獻
* [Wikipedia:Lua](http://en.wikipedia.org/wiki/Lua_programming_language)
*  Lua 5.2 Reference Manual -- <http://www.lua.org/manual/5.2/manual.html>
* <http://stackoverflow.com/questions/16641826/how-do-i-build-lua-for-windows-using-mingw-and-msys>


【本文由陳鍾誠取材並修改自 [維基百科]，採用創作共用的 [姓名標示、相同方式分享] 授權】
