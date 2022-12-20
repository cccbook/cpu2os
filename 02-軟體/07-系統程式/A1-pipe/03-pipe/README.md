# 匿名管道

使用 pipe({fd1,fd2}) 可以創建匿名管道功能！

(但通常是在 fork 的 process 之間使用！)

pipe({readfd,writefd})

## pipe0.c

```
mac020:03-pipe mac020$ ./pipe1 hello
child: hello
mac020:03-pipe mac020$ ./pipe1 'Hello how are you'
child: Hello how are you
```