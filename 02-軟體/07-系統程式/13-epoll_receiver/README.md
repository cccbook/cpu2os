# tcp-thread chat


## 編譯

```
$ gcc server.c -o server -lpthread
$ gcc client.c -o client -lpthread
```

## 執行方法

必須開三個視窗，其中一個執行 server 8086，另外兩個是 client

Server

```
guest@localhost:~/sp/09-linux/c/02-multiroom/04-epoll$ ./server 8886
port=8886
snoopy: hi
ccc: hello
ccc: i am snoopy
snoopy: nice to meet you , snoopy
snoopy: how are you?
ccc: fine , and you ?
```

Client ccc

```
guest@localhost:~/sp/09-linux/c/02-multiroom/04-epoll$ ./client ccc 127.0.0.1 8886
snoopy: hi
hello
i am snoopy
snoopy: nice to meet you , snoopy
snoopy: how are you?
fine , and you ?
```

Client snoopy

```
uest@localhost:~/sp/09-linux/c/02-multiroom/04-epoll$ ./client snoopy 127.0.0.1 8886
hi
ccc: hello
ccc: i am snoopy
nice to meet you , snoopy
how are you?
ccc: fine , and you ?
```
