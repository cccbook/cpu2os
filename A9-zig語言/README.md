# zig

* [zig](https://en.wikipedia.org/wiki/Zig_(programming_language))
    * Since version 0.10 the (new default) Zig compiler is written in Zig
    * The default backend (i.e. the optimizer) is still LLVM
    * Bun is a JavaScript and TypeScript runtime written in Zig, using Safari's JavaScriptCore virtual machine.

## 指令用法

```
$ zig
info: Usage: zig [command] [options]

Commands:

  build            Build project from build.zig
  fetch            Copy a package into global cache and print its hash
  init-exe         Initialize a `zig build` application in the cwd
  init-lib         Initialize a `zig build` library in the cwd

  ast-check        Look for simple compile errors in any set of files
  build-exe        Create executable from source or object files
  build-lib        Create library from source or object files
  build-obj        Create object from source or object files
  fmt              Reformat Zig source into canonical form
  run              Create executable and run immediately (常用)
  test             Create and run a test build (常用)
  translate-c      Convert C code to Zig code (讚)

  ar               Use Zig as a drop-in archiver
  cc               Use Zig as a drop-in C compiler
  c++              Use Zig as a drop-in C++ compiler
  dlltool          Use Zig as a drop-in dlltool.exe
  lib              Use Zig as a drop-in lib.exe
  ranlib           Use Zig as a drop-in ranlib
  objcopy          Use Zig as a drop-in objcopy
  rc               Use Zig as a drop-in rc.exe

  env              Print lib path, std path, cache directory, and version
  help             Print this help and exit
  libc             Display native libc paths file or validate one
  targets          List available compilation targets
  version          Print version number and exit
  zen              Print Zen of Zig and exit

General Options:

  -h, --help       Print command-specific usage

error: expected command argument
```

## Learn

* https://ziglearn.org/
    * https://github.com/Sobeston/ziglearn
    * https://zighelp.org/
* [A half-hour to learn Zig](https://gist.github.com/ityonemo/769532c2017ed9143f3571e5ac104e50)
* https://ziglang.org/learn/why_zig_rust_d_cpp
* https://codeberg.org/ziglings/exercises/
* https://github.com/belse-de/zig-tut

## package

* https://zig.pm/ (類似 Python 的 PyPi)
    * https://github.com/ziglibs/repository

## Resource

* https://project-awesome.org/catdevnull/awesome-zig
* https://github.com/C-BJ/awesome-zig