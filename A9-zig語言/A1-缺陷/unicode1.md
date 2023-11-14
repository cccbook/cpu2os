# zig 的中文問題

```
PS D:\ccc\ccc112a\cpu2os\A9-zig語言\03-lang
> zig run unicode1.zig
D:\install\zig-windows-x86_64\lib\std\fmt.zig:604:25: error: cannot format array ref without a specifier (i.e. {s} or {*})
PS D:\ccc\ccc112a\cpu2os\A9-zig語言\03-lang
> zig run unicode1.zig
*const [5:0]u8
5
e
0
Hello
中文
文
�
中文是否支援得好呢？
�支援得好呢？
��好呢？
��？
�
中文�文��是否支援得好呢？
援得好呢？
好呢？
��？
�
```

