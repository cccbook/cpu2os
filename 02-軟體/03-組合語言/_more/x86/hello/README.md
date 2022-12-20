# hello

* [如何在一台没有操作系统的裸机上显示一行字符？](https://www.zhihu.com/question/38424683/answer/76330253)


```
$ nasm boot.asm -o boot.bin
$ dd if=boot.bin of=boot.img bs=512 count=1 conv=notrunc
```

这里生成一个叫boot.img的软盘映像,再把它塞到虚拟机里,它会显示:
