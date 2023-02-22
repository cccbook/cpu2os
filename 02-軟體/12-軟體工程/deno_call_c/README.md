# deno call c

* https://deno.land/manual@v1.18.2/runtime/ffi_api

## windows + gcc(MinGW)

```
$ cc -c -o add_numbers.o add_numbers.c
$ cc -shared -Wall -o add_numbers.so add_numbers.o
$ deno run --allow-ffi --unstable ffi.ts
Check file:///C:/ccc/course/sp/11-se/deno_call_c/ffi.ts
Result from external addition of 35 and 34: 69
$ deno run --allow-ffi --unstable ffi_win_gcc.ts
Check file:///C:/ccc/course/sp/11-se/deno_call_c/ffi_win_gcc.ts
Result from external addition of 35 and 34: 69
```

## wsl

```
wsl> cc -c -o add_numbers.o add_numbers.c
wsl> cc -shared -Wall -o add_numbers.so add_numbers.o
wsl> deno run --allow-ffi --unstable ffi.ts
Check file:///mnt/wsl/docker-desktop-bind-mounts/Ubuntu-20.04/fdbd7e23af76ca956070b6eadda658eaefba4c755edccae5481002971e28d70b/course/sp/11-se/deno_call_c/ffi.ts
Result from external addition of 35 and 34: 69
```

