# room - poll

## 編譯

```
$ gcc server.c -o server -lpthread
$ gcc client.c -o client -lpthread
```

## 執行方法

必須開三個視窗，其中一個執行 server 8088，另外兩個是 client

Server

```
mac020:03-poll mac020$ ./server 8887
port=8887
snoopy: hi
ccc: hello
ccc: i am snoopy
snoopy: nice to meet you
snoopy: how are you ?
```

Client ccc

```
mac020:03-poll mac020$ ./client ccc 127.0.0.1 8887
snoopy: hi
hello
i am snoopy
snoopy: nice to meet you
snoopy: how are you ?
```

Client snoopy

```
mac020:03-poll mac020$ ./client snoopy 127.0.0.1 8887
hi
ccc: hello
ccc: i am snoopy
nice to meet you
how are you ?
```
