# Lua 語言簡介

Lua 是一個輕量級的程式語言，該語言是由巴西的 Computer Graphics Technology Group (Tecgraf) 這家公司的員工 Roberto Ierusalimschy, Luiz Henrique de Figueiredo, and Waldemar Celes 等人所創造的。

讀者可能會感到驚訝，為何會有一個來自巴西的程式語言呢？關於這件事，其實是有歷史典故的。很多技術的創新，其實都來自於某個歷史意外，Lua 語言的發明也是如此。

在 1977 年到 1992 年之間，巴西對電腦軟硬體進行了嚴格的貿易管制保護措施，很多需要電腦軟硬體的公司無法買到國外的軟硬體，於是 Tecgraf 這家主攻「電腦繪圖與使用者介面」的公司就只好自行發展「程式語言與開發工具」，並且提供給客戶使用。

在設計出 Lua 之前，Tecgraf 其實已經創造過 SOL (Simple Object Language) 與 DEL (data-entry language) 這兩個程式語言，但這兩個語言並沒有甚麼「流程控制」的語法，於是他們吸取了 SOL 與 DEL 的設計經驗，然後再參考了 「Modula 語言的 控制結構 (if, while, repeat/until)」，以及 「CLU 語言的 (多重指定、多傳回值) 等特性」，還有 「SNOBOL、AWK 中的字典 (associative arrays) 概念」，最後創造出了一種語法非常簡易的語言，那就是 Lua。

為何叫做 Lua 呢？Lua 一詞是葡萄牙語中「Luna」（月亮）的意思，因為其前身 SOL 語言的葡萄牙語意義為「太陽」，所以下一個語言就自然叫做「月亮」囉！

Lua 的語法非常的簡單，其 BNF 語法寫起來只有一頁，因此可以說是世界上最簡單的語言之一 (雖然 LISP 的語法比 Lua 更簡單，其 BNF 應該只要兩三行就寫完了，但是寫起來卻很不容易閱讀；而 Lua 則是容易讀容易寫，但是 BNF 語法就比 LISP 稍長了一些)。

由於這種簡易的特性，後來 Lua 被廣泛使用在各種遊戲引擎裏做為「腳本語言」(Script Language) 使用，列表如下：

* 2D 遊戲引擎 : Corona SDK、Love2D、Agen、Blitwizard、Cocos2d-x、EGSL、Grail、MOAI。
* 2.5D 遊戲引擎 : Lavgine。
* 3D 遊戲引擎 : Baja、Glint 3d、Irrlicht、Leadwerks、Spring RTS、Luxinia、Polycode、Cryengine。

另外、Lua 也常被各式各樣需要腳本語言的工具軟體使用，像是 Damn Small Linux、Wikipedia template engine、Wireshark、Nginx、LuaTeX、Adobe Photoshop Lightroom 等軟體裏都有內嵌 Lua 腳本語言。

官方版 Lua 預設的編譯器是用 C 語言寫成的，不過也有其他人寫出用 Lua 寫的 Lua 解譯器與編譯器，以下是相關資訊的網址：

* <http://lua-users.org/wiki/LuaInterpreterInLua>
* <http://lua-users.org/wiki/LuaCompilerInLua>

另外、也有像 LuaJIT 與 LLVM-Lua 這樣的立即編譯引擎 (Just in Time Compiler)，以下是其網址：

* <http://luajit.org/>
* <https://code.google.com/p/llvm-lua/>

而且、在 LuaWiki 這個網站中，也有不少關於 Lua 原始碼與實作的研究，您可以從下列網址找到：

* <http://lua-users.org/wiki/LuaSource>
* <http://lua-users.org/wiki/LuaImplementations>
* <http://lua-users.org/wiki/LuaGrammar>

如果您用過 JavaScript，應該會覺得這個語言怎麼這麼簡單又強大。但是當您看過 Lua 之後，可能會覺得 JavaScript 還是有點複雜，因為 Lua 感覺就像是「脫光光」的 JavaScript，沒有任何一點累贅的語法啊！

## 參考文獻

* [Wikipedia:Lua](http://en.wikipedia.org/wiki/Lua_programming_language)
* TecGraf -- <http://lac-rio.com/tecgraf>
* <http://stackoverflow.com/questions/5053134/what-is-a-good-game-engine-that-uses-lua>


【本文由陳鍾誠取材並修改自 [維基百科]，採用創作共用的 [姓名標示、相同方式分享] 授權】
