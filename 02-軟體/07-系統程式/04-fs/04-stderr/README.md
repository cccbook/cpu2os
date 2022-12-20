# 將錯誤訊息 stderr 導入檔案

## stderr1.c

```
mac020:04-stderr mac020$ gcc stderr1.c -o stderr1
mac020:04-stderr mac020$ ./stderr1
Warning: xxx
Error: yyy
```

## stderr2.c

```
mac020:04-stderr mac020$ gcc stderr2.c -o stderr2
mac020:04-stderr mac020$ ./stderr2
mac020:04-stderr mac020$ cat log.txt
Warning: xxx
Error: yyy
```
