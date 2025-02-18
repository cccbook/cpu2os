## Lua 引擎的解析與使用

本文解析的是 Lua 5.2.3 的原始碼，筆者採用 MinGW 在 MS. Windows 7 上進行編譯測試。

### 基本型態

Lua 是一種弱型態 (Weakly Typed) 的腳本語言 (Script Language)，就像 JavaScript/Python/Ruby/Perl 一樣，您不需像在 C/C++/Java 等語言中那樣宣告變數型態，但每個變數值其實都有個對應的型態，因為 Lua 會自動決定其型態。

Lua 中的基本型態如下：

型態                | 說明
--------------------|--------------------------------------------------------------------------
string              | 字串：8 bit 字元、以 \0 結尾。
number              | 數字：以 C 的 double 為預設型態，可代表正負整數與浮點數等。
nil                 | 空值：類似 C 語言的 NULL。
boolean             | 布林：真或假。
function            | 函數：與 javascript 中的函數一樣，都是第一級變數，非次等公民。
userdata            | 使用者定義資料：任何 C 語言中的資料都可以儲存成 Lua 的 userdata 變數。
table               | 表格：使用字典 associative array 結構，採用 (key, value) 配對結構。
thread              | 線程 (執行緒)：執行的基本程式單元。

上述的 table 結構威力強大， userdata 也可封裝成類似 table 的 (key,value) 結構，這種結構可用來表示「樹 (Tree)、圖 (Graph)、甚至是 XML」 等複雜結構。

Lua 的「物件導向」也是建構在 table 這樣的 (key, value) 結構之上的，依靠的是一種稱為 Meta-Methods 的概念。您可以將對應到物件裏的某欄位的 (key, value) 結構，也就是 (欄位、值) 的「值」改掉，就能完成像「抽換、繼承、多型、依賴性注射」之類的慨念。

Lua 的函數採用傳參照 (call by reference) 的方式，但是卻不允許參數在函數內被修改，還好 Lua 的函數可以傳回多值結果，因此任何被修改的參數都可以透過多值回傳的方式傳回。

### Lua 內部的運作方式

由於 Lua 強調可內嵌於其它應用程式的功能，因此允許隨時「解譯、載入、執行」，而且應用程式可以輕易的「捕捉錯誤」，甚至同時執行很多個 Lua 程式等等。

當您想載入 Lua 程式並執行時，通常會經過下列過程：

1. 透過 `luaL_newstate()` 創造出一份「狀態物件」(State)，這個狀態物件代表 Lua 語言。
2. 透過 `luaL_openlibs()` 載入原本的系統函數，或進一步註冊您自訂的 Lua 延伸函數。
3. 透過 `luaL_loadfile()` 載入 Lua 的程式碼，這些程式碼會被轉換成可被虛擬機執行的中間碼 (IR, bytecode)。
4. 透過 `lua_pcall()` 會呼叫虛擬機去執行 Lua 的中間碼 (bytecode)，若執行有錯就會傳回 1，否則會傳回 0。

在上述步驟中，luaL_loadfile() 這個函數會進行「掃描 lex、剖析 parse、編譯、compile」等動作，這些動作採用 「Pipes & Filters」的方式進行。

目前的 Lua 引擎並不採用像 lex 或 yacc 這樣的工具來產生剖析程式碼，而是採用手工打造的剖析器，來進行剖析與中間碼產生的動作。

當您呼叫 lua_pcall() 函數的時候，其實是呼叫虛擬機去執行 Lua 的中間碼 (bytecode)，虛擬機會解譯每個中間碼指令並執行對應的動作。

### 建置 Lua 解譯器 -- myLua

那麼、要怎麼用 C 呼叫並載入並執行一段「Lua 程式」呢？以下是 lua 原始碼中的解譯器 lua.c 裏的主程式，您可以用這種方法載入並執行 lua 程式。

```CPP
int main (int argc, char **argv) {
  int status, result;
  lua_State *L = luaL_newstate();  /* 步驟1. create state */
  if (L == NULL) {
    l_message(argv[0], "cannot create state: not enough memory");
    return EXIT_FAILURE;
  }
  // 步驟2. 您可以透過註冊函數 pushfunction(L, &func) 的方式註冊延伸函數。
  /* call 'pmain' in protected mode */
  lua_pushcfunction(L, &pmain);
  lua_pushinteger(L, argc);  /* 1st argument */
  lua_pushlightuserdata(L, argv); /* 2nd argument */
  status = lua_pcall(L, 2, 1, 0);
  result = lua_toboolean(L, -1);  /* get result */
  finalreport(L, status);
  lua_close(L);
  return (result && status == LUA_OK) ? EXIT_SUCCESS : EXIT_FAILURE;
}
```

以下是一個簡化過的 Lua 解譯器程式：

檔案：myLua.c

```CPP
#include <stdlib.h>
#include <stdio.h>

#include "lua.h"
#include "lauxlib.h"
#include "lualib.h"

void error(lua_State *L, char *msg){
	fprintf(stderr, "\nFATAL ERROR:\n  %s: %s\n\n", msg, lua_tostring(L, -1));
	exit(1);
}

int main(int argc, char *argv[])
{
    lua_State *L = luaL_newstate();             // 1. 建立 State 狀態機
    luaL_openlibs(L);                           // 2. 載入 Lua 基本函式庫
	
    if (luaL_loadfile(L, argv[1]))              // 3. 載入 argv[1] 指定的 Lua 程式檔 (還沒執行)
	  error(L, "luaL_loadfile() failed");      

    if (lua_pcall(L, 0, 0, 0))                  // 4. 執行載入的 Lua 程式
	  error(L, "lua_pcall() failed");

    lua_close(L);                               // 5. 結束
    return 0;
}

```

然後我們修改 Lua 5.2.3 版原始碼的 Makefile 如下 (其中的 CC=gcc, MYOBJS=..., myLua: ..., myLuac:... 等段落是我們加入的)：

```
CC=gcc
OBJS0=lapi.o lcode.o lctype.o ldebug.o ldo.o ldump.o lfunc.o lgc.o llex.o lmem.o lobject.o lopcodes.o lparser.o lstate.o lstring.o ltable.o ltm.o lundump.o lvm.o lzio.o
OBJS= $(OBJS0) lua.o lauxlib.o lbaselib.o lbitlib.o lcorolib.o ldblib.o liolib.o lmathlib.o loslib.o lstrlib.o ltablib.o loadlib.o linit.o
OBJS2= $(OBJS0) luac.o lauxlib.o

MYOBJS= $(OBJS0) myLua.o lauxlib.o lbaselib.o lbitlib.o lcorolib.o ldblib.o liolib.o lmathlib.o loslib.o lstrlib.o ltablib.o loadlib.o linit.o
MYOBJS2= $(OBJS0) myLuac.o lauxlib.o


CFLAGS= -Wall -O2

T= lua

all:	$T luac
	./$T test.lua
	./luac -l test.lua

$T:	$(OBJS)
	$(CC) -o $@ $(OBJS) -lm

luac:	$(OBJS2)
	$(CC) -o $@ $(OBJS2) -lm

myLua:	$(MYOBJS)
	$(CC) -o $@ $(MYOBJS) -lm

myLuac:	$(MYOBJS2)
	$(CC) -o $@ $(MYOBJS2) -lm
	
clean:
	rm -f $T $(OBJS) $(OBJS2) core core.* luac.out luac

diff:
	diff ORIG . | grep -v ^Only > DIFFS
```

接著我們就可以進行編譯，以下是筆者的操作過程：

```
ccc@ccc-PC /c/lua
$ make myLua
gcc -Wall -O2   -c -o myLua.o myLua.c
gcc -o myLua lapi.o lcode.o lctype.o ldebug.o ldo.o ldump.o lfunc.o lgc.o llex.o
 lmem.o lobject.o lopcodes.o lparser.o lstate.o lstring.o ltable.o ltm.o lundump
.o lvm.o lzio.o myLua.o lauxlib.o lbaselib.o lbitlib.o lcorolib.o ldblib.o lioli
b.o lmathlib.o loslib.o lstrlib.o ltablib.o loadlib.o linit.o -lm

ccc@ccc-PC /c/lua
$ ls *.lua
fact.lua  hello.lua  obj.lua  test.lua

ccc@ccc-PC /c/lua
$ myLua hello.lua
Hello World!

ccc@ccc-PC /c/lua
$ myLua fact.lua
factorial(5)=120

ccc@ccc-PC /c/lua
$ myLua obj.lua
point(10,20)
point(30,40)
point(50,60)

ccc@ccc-PC /c/lua
$ myLua xxx.lua

FATAL ERROR:
  luaL_loadfile() failed: cannot open xxx.lua: No such file or directory
```

### 建置 Lua 編譯器 -- myLuac

由於 Lua 定義有中間碼格式，因此也可以將該格式輸出，變成 bytecode。

Lua 環境中的 luac 這個程式就是 Lua 編譯器，可以將 Lua 程式轉換成 bytecode ，但是若我們想在 C 語言裏寫程式將 Lua 程式轉為 bytecode 該怎麼做呢？以下是一個簡易的範例。

檔案：myLuac.c

```CPP
#include <stdlib.h>
#include <stdio.h>

#include "lua.h"
#include "lauxlib.h"
#include "lualib.h"
#include "lundump.h"
#include "lobject.h"
#include "lstate.h"

#define toproto(L,i) getproto(L->top+(i))

static int writer(lua_State* L, const void* p, size_t size, void* u)
{
 UNUSED(L);
 return (fwrite(p,size,1,(FILE*)u)!=1) && (size!=0);
}

void error(lua_State *L, char *msg){
	fprintf(stderr, "\nFATAL ERROR:\n  %s: %s\n\n", msg, lua_tostring(L, -1));
	exit(1);
}

int main(int argc, char *argv[])
{
    lua_State *L = luaL_newstate();             // 1. 建立 State 狀態機
	
    if (luaL_loadfile(L, argv[1]))              // 2. 載入 argv[1] 指定的 Lua 程式檔 (還沒執行)
	  error(L, "luaL_loadfile() failed");      

	int stripping=0;
	Proto* f = (Proto*) toproto(L,-1);
    FILE* D= fopen(argv[2],"wb");
	if (D != NULL) {
      lua_lock(L);
      luaU_dump(L,f,writer,D,stripping);        // 3. 輸出編譯後的檔案
      lua_unlock(L);
	}
    lua_close(L);                               // 4. 結束
    return 0;
}
```

在上述的 Makefile 中，我們已經加入了 myLuac 的編譯指引，因此可以直接編譯了：

```
ccc@ccc-PC /c/lua
$ make myLuac
gcc -Wall -O2   -c -o myLuac.o myLuac.c
gcc -o myLuac lapi.o lcode.o lctype.o ldebug.o ldo.o ldump.o lfunc.o lgc.o llex.
o lmem.o lobject.o lopcodes.o lparser.o lstate.o lstring.o ltable.o ltm.o lundum
p.o lvm.o lzio.o myLuac.o lauxlib.o -lm

ccc@ccc-PC /c/lua
$ myLua hello.lua hello.lo
Hello World!

ccc@ccc-PC /c/lua
$ myLuac hello.lua hello.lo

ccc@ccc-PC /c/lua
$ myLua hello.lo
Hello World!

ccc@ccc-PC /c/lua
$ myLuac fact.lua fact.lo

ccc@ccc-PC /c/lua
$ myLua fact.lo
factorial(5)=120
```

上述執行過程的最後一個指令，是我們用 myLua 去執行 myLuac 編譯出來的 bytecode，所得到的結果。

您可以看到，myLua 除了執行像 fact.lua, hello.lua 的原始程式檔之外，也可以執行編譯後的 bytecode，像是上述的 fact.lo, hello.lo 等等中間檔。

### 結語

在本文中，我們介紹了 Lua 環境的基本架構，並修改建置檔 Makefile，建立了一個簡易的的 Lua 解譯器程式，透過這樣的方式，讀者應該已經掌握了 Lua 解譯器的建置、連結與執行方式，並且對 Lua 的運作過程有一定的瞭解了。

### 參考文獻
* [Lua Source](http://lua-users.org/wiki/LuaSource)
* [The Lua Architecture - Advanced Topics in Software Engineering (DOC)](http://pgl.yoyo.org/lua/docs/luaarchitecture.doc), Mark Stroetzel Glasberg, Jim Bresler, Yongmin Kevin Cho.
* [Calling Lua From a C Program (With Snippets)](http://www.troubleshooters.com/codecorn/lua/lua_c_calls_lua.htm)