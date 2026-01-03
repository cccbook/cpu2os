這版是 Grok 寫的

但顯然還沒完成，產生的 vm 檔太過簡單，應該是錯的 ....

請看 Seven/Main.jack

```
class Main {

   function void main() {
       do Output.printInt(1 + (2 * 3));
       return;
   }

}

```

只產生出

```
function Main.main 0
push constant 0
return

```

整個

```
do Output.printInt(1 + (2 * 3));
```

語句完全被忽略

11py 中有 haviva 的 python 版本

執行結果如下

```
function Main.main 0
push constant 1
push constant 2
push constant 3
call Math.multiply 2  
add  
call Output.printInt 1
pop temp 0
push constant 0
return  

```

看來是因為 compile_subroutine 裡面沒有處理 do 語句

以上案例已解決 ...

但是最後的 

```
push constant 0
return  
```

還沒有輸出

所以 return 沒有處理好 （是因為第二個語句的關係嗎？）

另外 while 迴圈也沒處理

