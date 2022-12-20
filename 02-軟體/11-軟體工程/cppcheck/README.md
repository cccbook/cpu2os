# cppcheck

* [程式開發 — 使用Cppcheck做靜態程式碼分析](https://medium.com/@sepfy95/%E7%A8%8B%E5%BC%8F%E9%96%8B%E7%99%BC-%E4%BD%BF%E7%94%A8cppcheck%E5%81%9A%E9%9D%9C%E6%85%8B%E7%A8%8B%E5%BC%8F%E7%A2%BC%E5%88%86%E6%9E%90-8ccc4b1b4462)
* [cppcheck 靜態程式分析工具](https://ivan7645.github.io/2016/07/01/linux_cppcheck/)
* [使用Cppcheck靜態分析工具輔助檢查C++程式潛在問題](https://dotblogs.com.tw/larrynung/2011/10/29/47866)

* https://hackmd.io/@sysprog/linux2021-lab0

* https://cppcheck.sourceforge.io/


## 使用

```
$ wsl
wsl> cppcheck alloc.c --enable=all
Checking alloc.c ...
alloc.c:6:1: error: Memory leak: p [memleak]
}
^
nofile:0:0: information: Cppcheck cannot find all the include files (use --check-config for details) [missingIncludeSystem]

wsl> cppcheck alloc1.c --enable=all
Checking alloc1.c ...
alloc1.c:6:1: error: Memory leak: p [memleak]
}
^
nofile:0:0: information: Cppcheck cannot find all the include files (use --check-config for details) [missingIncludeSystem]

wsl> cppcheck alloc2.c --enable=all
Checking alloc2.c ...
nofile:0:0: information: Cppcheck cannot find all the include files (use --check-config for details) [missingIncludeSystem]
```