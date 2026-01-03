# tinysql 專案

來源 -- https://github.com/shxntanu/tinysql

## 執行結果

```
(py310) cccimac@cccimacdeiMac 00-tinysql % cd tinysql
(py310) cccimac@cccimacdeiMac tinysql % pytest test
========================= test session starts ==========================
platform darwin -- Python 3.10.16, pytest-8.3.4, pluggy-1.5.0
rootdir: /Users/cccimac/Desktop/ccc/cpu2os/02-系統程式/A1-資料庫/00-tinysql/tinysql
plugins: anyio-4.9.0
collected 2 items                                                      

test/test_insert.py ..                                           [100%]

========================== 2 passed in 0.01s ===========================
(py310) cccimac@cccimacdeiMac tinysql % ls
assets          Makefile        README.md       test
CMakeLists.txt  mydb.db         src             tinysql
(py310) cccimac@cccimacdeiMac tinysql % pwd
/Users/cccimac/Desktop/ccc/cpu2os/02-系統程式/A1-資料庫/00-tinysql/tinysql
(py310) cccimac@cccimacdeiMac tinysql % make clean
rm -f db *.db
(py310) cccimac@cccimacdeiMac tinysql % make
gcc src/*.c -o tinysql
(py310) cccimac@cccimacdeiMac tinysql % make test
gcc src/*.c -o tinysql
python3 -m unittest test.test_insert
..
----------------------------------------------------------------------
Ran 2 tests in 0.384s

OK
```

## 研究

* 程式說明: 
    * 對話 -- https://gemini.google.com/app/7e0b5269f7d529c5
    * 分享 -- https://gemini.google.com/share/87ae3ebcf7a9

