# Lua 的程式範例

本文的 Lua 程式乃是在 MS. Windows 的 MinGW 環境下執行的結果。

檔案：hello.lua

```Lua
print('Hello World!')
```

執行結果：

```
ccc@ccc-PC /c/lua
$ lua hello.lua
Hello World!
```

範例：註解語法

```Lua
-- A comment in Lua starts with a double-hyphen and runs to the end of the line.
 
--[[ Multi-line strings & comments
     are adorned with double square brackets. ]]
 
--[=[ Comments like this can have other --[[comments]] nested. ]=]
```

範例：迴圈語法

```Lua
--condition = true
 
while condition do
  --statements
end
 
repeat
  --statements
until condition
 
for i = first,last,delta do     --delta may be negative, allowing the for loop to count down or up
  print(i)
end

for key, value in pairs(_G) do
  print(key, value)
end
```

檔案：fact.lua -- 計算階層 n! 

```Lua
function factorial(n)
  local x = 1
  for i = 2,n do
    x = x * i
  end
  return x
end

print("factorial(5)="..factorial(5));
```

執行結果：

```
ccc@ccc-PC /c/lua
$ lua fact.lua
factorial(5)=120
```

檔案：obj.lua -- 物件

```Lua
function Point(x, y)        -- "Point" object constructor
  return { x = x, y = y }   -- Creates and returns a new object (table)
end

array = { Point(10, 20), Point(30, 40), Point(50, 60) }   -- Creates array of points

for i = 1,3 do 
  print("point("..array[i].x..","..array[i].y..")")
end
```

執行結果：

```
ccc@ccc-PC /c/lua
$ lua obj.lua
point(10,20)
point(30,40)
point(50,60)
```

另外、Lua 不只可以直接執行，也可以先編譯成中間碼 (bytecode) 之後再執行，如下所示：

```
ccc@ccc-PC /c/lua
$ luac -o obj.lo obj.lua

ccc@ccc-PC /c/lua
$ lua obj.lo
point(10,20)
point(30,40)
point(50,60)
```


## 參考文獻
* [Wikipedia:Lua](http://en.wikipedia.org/wiki/Lua_programming_language)

【本文由陳鍾誠取材並修改自 [維基百科]，採用創作共用的 [姓名標示、相同方式分享] 授權】