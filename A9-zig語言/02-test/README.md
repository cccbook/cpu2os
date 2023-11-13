# test

## iftest.zig

```
$ zig test iftest.zig
All 1 tests passed.
```

## iftest_bug.zig

```
$ zig test iftest_bug.zig
Test [1/1] test.if statement... FAIL (TestUnexpectedResult)
D:\install\zig-windows-x86_64\lib\std\testing.zig:527:14: 0x7ff6515e102f in expect (test.exe.obj)
    if (!ok) return error.TestUnexpectedResult;
             ^
D:\ccc\code\zig\02\iftest_bug.zig:11:5: 0x7ff6515e1195 in test.if statement (test.exe.obj)
    try expect(x == 2);
    ^
0 passed; 0 skipped; 1 failed.
error: the following test command failed with exit code 1:
C:\Users\user\AppData\Local\zig\o\95a0a4f03952e57a4b4e4b427c1129e9\test.exe
```
