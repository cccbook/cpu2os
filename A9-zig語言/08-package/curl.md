# curl.zig

## run

應該要在 linux/msys2/wsl/mac ...

一般 windows 不可以

```
$ zig build-exe zig-curl-test.zig --library curl --library c $(pkg-config --cflags libcurl)
```

