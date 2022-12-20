# 具名管道

使用 mkfifo 創建檔案後，就可以用 read, write 當成管道使用！

## mkfifo

```
mac020:04-fifo mac020$ mkfifo myfifo
mac020:04-fifo mac020$ ls
README.md       fifo1.c         myfifo
mac020:04-fifo mac020$ ls -all myfifo
prw-r--r--  1 mac020  staff  0  5 25 10:45 myfifo
```

## use

```
mac020:04-fifo mac020$ ls -all myfifo
prw-r--r--  1 mac020  staff  0  5 25 10:45 myfifo
mac020:04-fifo mac020$ cat < myfifo
^C
mac020:04-fifo mac020$ cat < myfifo&
[1] 31102
mac020:04-fifo mac020$ echo "hello myfifo" > myfifo
mac020:04-fifo mac020$ hello myfifo

[1]+  Done                    cat < myfifo
```
