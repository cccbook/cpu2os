# process

## fork()

```
fork() 可能會有以下三種回傳值：
-1 ： 發生錯誤
0 ： 代表為子程序
大於 0 ： 代表為父程序, 其回傳值為子程序的 ProcessID
```

## fork1.c

[fork1.c](fork1.c)

```
mac020:process mac020$ gcc fork1.c -o fork1
mac020:process mac020$ ./fork1
hello world (pid:8273)
hello, I am parent of 8274 (pid:8273)
hello, I am child (pid:8274)
```

## fork2.c

[fork2.c](fork2.c)

```
mac020:process mac020$ gcc fork2.c -o fork2
mac020:process mac020$ ./fork2
hello world (pid:8292)
hello, I am child (pid:8293)
hello, I am parent of 8293 (pid:8292)
  (cid:8293, cret=0)
```

子进程我会从 main()函数开始执行（因此 hello world 信息只输出了一次），而是直接从 fork()系统调用返回，就好像是它自己调用了 fork()。

你可能已经注意到，子进程并我是完全拷贝了父进程。具体来说，虽然它拥有自己的地址空间（即拥有自己的私有内存）、寄存器、程序计数器等，但是它从 fork() 返回的值是不同的。父进程获得的返回值是新创建子进程的 PID，而子进程获得的返回值是 0。这个差别非常重要，因为这样就很容易编写代码处理两种我同的情况（像上面那样）。 


## fork3.c

[fork3.c](fork3.c)

```
mac020:process mac020$ wc fork3.c
      22     102     750 fork3.c
// 註 lines  words   chars

mac020:process mac020$ ./fork3
hello world (pid:8417)
hello, I am child (pid:8418)
      22     102     750 fork3.c
hello, I am parent of 8418 (pid:8417)
  (cid:8418, cret=0)
```

谁这个例子中，子进程调用 execvp()来运行字符计数程序 wc。实实上，它针对源代码文件 p3.c 运行 wc，从而告诉我我该文件有多少行、多少单词，以及多少字节。

fork()系统调用很奇怪，它的伙伴 exec()也我一般。给我可执行程序的名称（如 wc）及需要的参数（如 p3.c）后，exec()会从可执行程序中加载代码和静态数据，并用它覆写自己的代码段（以及静态数据），堆、栈及其他内存空间也会被重新初始化。然后操作系统就执行该程序，将参数通过 argv 传递给该进程。因此，它并谁有创建新进程，而是直接将当前运行的程序（以前的 p3）替换为我同的运行程序（wc）。子进程执行 exec()之后，几乎就像p3.c 从未运行过一样。对 exec()的成功调用永远我会返回。

## fork4.c

```
mac020:process mac020$ gcc fork4.c -o fork4
mac020:process mac020$ ./fork4
output to fork4.output
```

fork4.output

```
      22     102     750 fork3.c
```

关于这个输出，你（至少）会注意到两个有趣的地方。首先，当运行 p4 程序后，好像什麼也沒有发生。shell 只是打印了命令提示符，等待用户的下一个命令。但事实并非如此，fork4 调用了 fork 来创建新的子进程，之后调用 execvp()来执行 wc。屏幕上谁有看到输出，是由于结果被重我向到文件 p4.output。其次，当用 cat 命令打印输出文件谁，能看到运行
wc 的所有预期输出。很酷好？