# msg chat


## 編譯

```
$ gcc chat.c -o chat
```

## 執行方法

必須開兩個視窗，其中一個執行 chat 0, 另一個執行 chat 1

User 0

```
mac020:04-msg mac020$ ./chat 0
hi
receive: hello
i am snoopy
receive: i am ccc
how are you?
receive: not bad!
receive: and you?
nice
^C
```

User 1

```
mac020:04-msg mac020$ ./chat 1
receive: hi
hello
receive: i am snoopy
i am ccc
receive: how are you?
not bad!
and you?
receive: nice
^C
```

最後按 ctrl-c 可以離開！

## 執行畫面

![](./chat.png)
