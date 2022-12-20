## windows+Visual Studio

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
-- Build files have been written to: C:/ccc/course/sp/01-gcc/06-cmake/04-shared-library/build
```

然後用 Visual Studio 建置，結果失敗，出現下列訊息

```
嚴重性	程式碼	說明	專案	檔案	行	隱藏項目狀態
錯誤	LNK1104	無法開啟檔案 'Debug\hello_library.lib'	hello_binary	C:\ccc\course\sp\01-gcc\06-cmake\04-shared-library\build\LINK	1	
```

## msys2

```
  355  mkdir build && cd build
  356  cmake ..
  357  ninja
  358  ./hello_binary




Hero3C@DESKTOP-O093POU MINGW64 /c/ccc/course/sa/c/package/cmake
$ cd D-shared-library/

Hero3C@DESKTOP-O093POU MINGW64 /c/ccc/course/sa/c/package/cmake/D-shared-library
$ ls
CMakeLists.txt  README.adoc  include  src

Hero3C@DESKTOP-O093POU MINGW64 /c/ccc/course/sa/c/package/cmake/D-shared-library
$ mkdir build && cd build

Hero3C@DESKTOP-O093POU MINGW64 /c/ccc/course/sa/c/package/cmake/D-shared-library/build
$ cmake ..
-- Building for: Ninja
-- The C compiler identification is GNU 10.2.0
-- The CXX compiler identification is GNU 10.2.0
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: C:/msys64/usr/bin/cc.exe - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: C:/msys64/usr/bin/c++.exe - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Configuring done
-- Generating done
-- Build files have been written to: C:/ccc/course/sa/c/package/cmake/D-shared-library/build

Hero3C@DESKTOP-O093POU MINGW64 /c/ccc/course/sa/c/package/cmake/D-shared-library/build
$ ls
CMakeCache.txt  CMakeFiles  build.ninja  cmake_install.cmake

Hero3C@DESKTOP-O093POU MINGW64 /c/ccc/course/sa/c/package/cmake/D-shared-library/build
$ ninja
[1/4] Building CXX object CMakeFiles/hello_binary.dir/src/main.cpp.obj
[2/4] Building CXX object CMakeFiles/hello_library.dir/src/Hello.cpp.obj
[3/4] Linking CXX shared library libhello_library.dll
[4/4] Linking CXX executable hello_binary.exe

Hero3C@DESKTOP-O093POU MINGW64 /c/ccc/course/sa/c/package/cmake/D-shared-library/build
$ ./hello_binary
Hello Shared Library!

Hero3C@DESKTOP-O093POU MINGW64 /c/ccc/course/sa/c/package/cmake/D-shared-library/build
$ ls
CMakeCache.txt  cmake_install.cmake   libhello_library.dll.a
CMakeFiles      hello_binary.exe
build.ninja     libhello_library.dll
```
