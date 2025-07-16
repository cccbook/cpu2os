# room - thread

## 編譯

```
$ gcc server.c -o server -lpthread
$ gcc client.c -o client -lpthread
```

## 執行方法

必須開三個視窗，其中一個執行 server 8088，另外兩個是 client

Server

```
mac020:02-multithread mac020$ ./server 8888
port=8888
ccc: hi
snoopy: hello
snoopy: i am snoopy

```

Client ccc

```
mac020:02-multithread mac020$ ./client ccc 127.0.0.1 8888
hi
snoopy: hello
snoopy: i am snoopy
```

Client snoopy

```
mac020$ ./client snoopy 127.0.0.1 8888
hello
i am snoopy

```
