# zig unicode

官方程式碼

* 測試 -- https://github.com/ziglang/zig/blob/master/lib/std/unicode/throughput_test.zig


* https://github.com/ziglang/zig/blob/master/lib/std/debug.zig

```zig
var stderr_mutex = std.Thread.Mutex{};

/// Print to stderr, unbuffered, and silently returning on failure. Intended
/// for use in "printf debugging." Use `std.log` functions for proper logging.
pub fn print(comptime fmt: []const u8, args: anytype) void {
    stderr_mutex.lock();
    defer stderr_mutex.unlock();
    const stderr = io.getStdErr().writer();
    nosuspend stderr.print(fmt, args) catch return;
}
```

* https://github.com/ziglang/zig/blob/master/lib/std/io/writer.zig

```
        pub fn print(self: Self, comptime format: []const u8, args: anytype) Error!void {
            return std.fmt.format(self, format, args);
        }
```

* https://github.com/ziglang/zig/blob/master/lib/std/fmt.zig#L22

```
pub fn format(....)
```


in powershell

```
> chcp 65001

Active code page: 65001
PS D:\ccc\ccc112a\cpu2os\A9-zig瑾炶█\03-lang
> zig run unicode.zig
*const [5:0]u8
5
e
0
true
128169
128175
鈿★拷锟�
涓拷锟�
true
true
0xfe
0x9f
PS D:\ccc\ccc112a\cpu2os\A9-zig瑾炶█\03-lang
> zig run unicode.zig
*const [5:0]u8
5
e
0
true
128169
128175
鈿★拷锟�
涓拷锟�
涓枃
鏂�
锟�
true
true
0xfe
0x9f
```

