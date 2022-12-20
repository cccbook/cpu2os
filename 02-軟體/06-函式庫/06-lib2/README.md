# lib2

## WSL

```
wsl> make
gcc -D__GNU__ -Wall -I../src/ std2_test.c ../src/std2.c -o std2_test
gcc -D__GNU__ -Wall -I../src/ gnu2_test.c ../src/std2.c -o gnu2_test
gcc -D__GNU__ -Wall -I../src/ f2_test.c ../src/std2.c -o f2_test
./std2_test
Check:one==1  
  pass!
Check:two==1  
  Error: at FILE=std2_test.c LINE=6
one=1
pi:str=3.141590
Error: new0 malloc() fail!
./gnu2_test
1.000000 2.000000 3.000000 4.000000 
2.000000 4.000000 6.000000 8.000000
3.000000 6.000000 9.000000 12.000000
./f2_test
Check:ARRAY_EQ(a, a, int, n)  
  pass!
1 4 9
Check:!ARRAY_EQ(a, b, int, n)  
  pass!
r=6
filter(a, odd)=1 3
Check:eveni==1  
  pass!
a:str=1 2 3
b:str=1 3 9
Check:EQ(1, 1)  
  pass!
Check:NE(1, 2)  
  pass!
Check:LT(1.3, 2.5)  
  pass!
Check:AND(true, true)  
  pass!
Check:OR(1, 0)  
  pass!
Check:NOT(false)  
  pass!
Check:NEG(10.0)<1.3  
  pass!
Check:SHL(1, 3)==8  
  pass!
Check:MUL(3, 7)==21  
  pass!
Check:n==3  
  pass!
Check:ALL(a, b, n, LT)  
  pass!
Check:ANY(a, b, n, LT)  
  pass!
Check:ANY(a, c, n, GT)  
  pass!
```

## CodeBlocks MINGW

```
$ make
gcc -D__GNU__ -Wall -I../src/ std2_test.c ../src/std2.c -o std2_test
gcc -D__GNU__ -Wall -I../src/ gnu2_test.c ../src/std2.c -o gnu2_test
gcc -D__GNU__ -Wall -I../src/ f2_test.c ../src/std2.c -o f2_test
./std2_test
Check:one==1  
  pass!
Check:two==1  
  Error: at FILE=std2_test.c LINE=6
one=1
pi:str=3.141590
Error: new0 malloc() fail!
./gnu2_test
1.000000 2.000000 3.000000 4.000000 
2.000000 4.000000 6.000000 8.000000
3.000000 6.000000 9.000000 12.000000
./f2_test
Check:ARRAY_EQ(a, a, int, n)  
  pass!
1 4 9
Check:!ARRAY_EQ(a, b, int, n)  
  pass!
r=6
filter(a, odd)=1 3
Check:eveni==1  
  pass!
a:str=1 2 3
b:str=1 3 9
Check:EQ(1, 1)  
  pass!
Check:NE(1, 2)  
  pass!
Check:LT(1.3, 2.5)  
  pass!
Check:AND(true, true)  
  pass!
Check:OR(1, 0)  
  pass!
Check:NOT(false)  
  pass!
Check:NEG(10.0)<1.3  
  pass!
Check:SHL(1, 3)==8  
  pass!
Check:MUL(3, 7)==21  
  pass!
Check:n==3  
  pass!
Check:ALL(a, b, n, LT)  
  pass!
Check:ANY(a, b, n, LT)  
  pass!
Check:ANY(a, c, n, GT)  
  pass!
```

## msys2 gcc

```
$ make clean
rm std2_test gnu2_test
$ make
gcc -D__GNU__ -Wall -I../src/ std2_test.c ../src/std2.c -o std2_test
gcc -D__GNU__ -Wall -I../src/ gnu2_test.c ../src/std2.c -o gnu2_test
./std2_test
Error: at FILE=std2_test.c LINE=12
  Check:two==1 but two==2

Error: at FILE=std2_test.c LINE=14
  Check:two==1 but two==2

1 2 3
1 4 9
2 6 12
./gnu2_test
1.000000 2.000000 3.000000 4.000000 
2.000000 4.000000 6.000000 8.000000
3.000000 6.000000 9.000000 12.000000
```

## clang

```
$ make clean
rm std2_test gnu2_test
$ make CC=clang
clang -Wall -I../src/ std2_test.c ../src/std2.c -o std2_test
./std2_test
Error: at FILE=std2_test.c LINE=12
  Check:two==1 but two==2

Error: at FILE=std2_test.c LINE=14
  Check:two==1 but two==2

1 2 3
1 4 9
2 6 12
```

## MS cl

```
PS C:\ccc\course\sp\07-lib\02-clib2\test> make clean
rm std2_test gnu2_test
rm: cannot remove 'gnu2_test': No such file or directory
make: *** [Makefile:30: clean] Error 1
PS C:\ccc\course\sp\07-lib\02-clib2\test> make CC=cl
cl -Wall -I../src/ std2_test.c ../src/std2.c -o std2_test
Microsoft (R) C/C++ Optimizing Compiler Version 19.29.30133 for x86
Copyright (C) Microsoft Corporation.  著作權所有，並保留一切權利。

cl : 命令列 warning D9035 : 選項 'o' 已被取代，在未來的發行版本中將會移除
std2_test.c
../src/std2.h(1): warning C4819: 檔案含有無法在目前字碼頁 (950) 中表示的字元。請以 Unicode 格式儲存檔案以防止資料遺失
std2_test.c(22): warning C4244: '=': 將 'double' 轉換為 'int'，由於類型不同，可能導致資料遺失
std2.c
C:\ccc\course\sp\07-lib\02-clib2\src\std2.h(1): warning C4819: 檔案含有無法在目前字碼頁 (950) 中表示的字元。請以 Unicode 格式儲存檔案以防止資料遺失
正在產生程式碼...
C:\ccc\course\sp\07-lib\02-clib2\test\std2_test.c(23) : warning C5045: 若指定了 /Qspectre 參數，編譯器將會為記憶體負載插入 Spectre 風險降低
C:\ccc\course\sp\07-lib\02-clib2\test\std2_test.c(23) : 注意: 對此行進行比較，以檢查索引 'i' 範圍
C:\ccc\course\sp\07-lib\02-clib2\test\std2_test.c(23) : 注意: 摘要此行上的呼叫
C:\ccc\course\sp\07-lib\02-clib2\test\std2_test.c(22) : warning C5045: 若指定了 /Qspectre 參數，編譯器將會為記憶體負載插入 Spectre 風險降低
C:\ccc\course\sp\07-lib\02-clib2\test\std2_test.c(22) : 注意: 對此行進行比較，以檢查索引 'i' 範圍
C:\ccc\course\sp\07-lib\02-clib2\test\std2_test.c(22) : 注意: 摘要此行上的呼叫
C:\ccc\course\sp\07-lib\02-clib2\test\std2_test.c(19) : warning C5045: 若指定了 /Qspectre 參數，編譯器將會為記憶體負載插入 Spectre 風險降低
C:\ccc\course\sp\07-lib\02-clib2\test\std2_test.c(19) : 注意: 對此行進行比較，以檢查索引 'i' 範圍
C:\ccc\course\sp\07-lib\02-clib2\test\std2_test.c(19) : 注意: 摘要此行上的呼叫
C:\ccc\course\sp\07-lib\02-clib2\test\std2_test.c(18) : warning C5045: 若指定了 /Qspectre 參數，編譯器將會為記憶體負載插入 Spectre 風險降低
C:\ccc\course\sp\07-lib\02-clib2\test\std2_test.c(18) : 注意: 對此行進行比較，以檢查索引 'i' 範圍
C:\ccc\course\sp\07-lib\02-clib2\test\std2_test.c(18) : 注意: 摘要此行上的呼叫
C:\ccc\course\sp\07-lib\02-clib2\test\std2_test.c(17) : warning C5045: 若指定了 /Qspectre 參數，編譯器將會為記憶體負載插入 Spectre 風險降低
C:\ccc\course\sp\07-lib\02-clib2\test\std2_test.c(17) : 注意: 對此行進行比較，以檢查索引 'i' 範圍
C:\ccc\course\sp\07-lib\02-clib2\test\std2_test.c(17) : 注意: 摘要此行上的呼叫
Microsoft (R) Incremental Linker Version 14.29.30133.0
Copyright (C) Microsoft Corporation.  All rights reserved.

/out:std2_test.exe
/out:std2_test.exe
std2_test.obj
std2.obj
./std2_test
[1;31mError: at FILE=std2_test.c LINE=12[0m
[1;32m  Check:two==1 but two==2
[0m
[1;31mError: at FILE=std2_test.c LINE=14[0m
[1;32m  Check:two==1 but two==2
[0m
1 2 3
1 4 9
2 6 12
PS C:\ccc\course\sp\07-lib\02-clib2\test>
```

## Reference

* lambda 語法
    * https://stackoverflow.com/questions/10405436/anonymous-functions-using-gcc-statement-expressions
    * https://github.com/wd5gnr/clambda/blob/master/clambda2.c
    * https://hackaday.com/2019/09/11/lambdas-for-c-sort-of/
    * https://gcc.gnu.org/onlinedocs/gcc/Statement-Exprs.html#Statement-Exprs
* https://shengwen1997.gitbooks.io/program_with_c/content/Stringification.html
* [使用泛型型別巨集 (_Generic) 撰寫泛型程式](https://opensourcedoc.com/c-programming/generics-with-generic-selection/)
