# bpTree (B+ Tree)

* 修改自 -- https://github.com/json-db/B-Plus-Tree

## 參考

* https://eatonphil.com/btrees.html

## Run

```
$ ./build.sh

$ ./a
[no.1   key = 10, value = 20]
[no.2   key = 10, value = 21]
Total number of answers is: 2
bptQueryKey(10)=2
[no.1   key = 10, value = 20]
[no.2   key = 10, value = 21]
[no.3   key = 11, value = 22]
[no.4   key = 11, value = 23]
[no.5   key = 12, value = 24]
[no.6   key = 12, value = 25]
[no.7   key = 13, value = 26]
[no.8   key = 13, value = 27]
[no.9   key = 14, value = 28]
[no.10  key = 14, value = 29]
[no.11  key = 15, value = 30]
[no.12  key = 15, value = 31]
[no.13  key = 16, value = 32]
[no.14  key = 16, value = 33]
[no.15  key = 17, value = 34]
[no.16  key = 17, value = 35]
[no.17  key = 18, value = 36]
[no.18  key = 18, value = 37]
[no.19  key = 19, value = 38]
...
Total number of answers is: 22
bptQueryRange(10,20)=22
Now destroying B+tree ..
```

## 參考

* [B-tree詳解及實現(C語言)](https://www.twblogs.net/a/5b8e068c2b7177188342564d)
* https://www.programiz.com/dsa/b-tree