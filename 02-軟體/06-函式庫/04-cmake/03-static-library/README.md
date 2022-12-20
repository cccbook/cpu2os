# cmake static library example

## windows + Visual Studio

```
$ mkdir build
$ cd build
$ cmake ..
-- Building for: Visual Studio 16 2019
-- Selecting Windows SDK version 10.0.19041.0 to target Windows 10.0.19043.
-- The C compiler identification is MSVC 19.29.30133.0
-- The CXX compiler identification is MSVC 19.29.30133.0
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Tools/MSVC/14.29.30133/bin/Hostx64/x64/cl.exe - skipped     
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Tools/MSVC/14.29.30133/bin/Hostx64/x64/cl.exe - skipped   
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Configuring done
-- Generating done
-- Build files have been written to: C:/ccc/course/sp/01-gcc/06-cmake/03-static-library/build
```


然後用 Visual Studio 開啟 build/hello_cmake.sln 並按下《建置/建置方案》去 build



```
$ cd Debug
$ ls
hello_binary.exe*  hello_binary.pdb  hello_library.lib  hello_library.pdb
$ ./hello_binary
Hello Static Library!
```