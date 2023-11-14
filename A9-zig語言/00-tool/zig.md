# zig

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
  run              Create executable and run immediately
  test             Create and run a test build
  translate-c      Convert C code to Zig code

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

## zig fetch 

```
$ zig fetch -h
Usage: zig fetch [options] <url>
Usage: zig fetch [options] <path>

    Copy a package into the global cache and print its hash.

Options:
  -h, --help                    Print this help and exit
  --global-cache-dir [path]     Override path to global Zig cache directory
  --debug-hash                  Print verbose hash information to stdout
```

## zig ranlib

```
$ zig ranlib -h
OVERVIEW: LLVM ranlib

Generate an index for archives

USAGE: ranlib archive...

OPTIONS:
  -h --help             - Display available options
  -v --version          - Display the version of this program
  -D                    - Use zero for timestamps and uids/gids (default)
  -U                    - Use actual timestamps and uids/gids
```




