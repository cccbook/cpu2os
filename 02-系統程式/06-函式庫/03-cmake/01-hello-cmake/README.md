## WSL

```
wsl> mkdir build
wsl> cd build
wsl> cmake ..
-- The C compiler identification is GNU 9.4.0
-- The CXX compiler identification is GNU 9.4.0
-- Check for working C compiler: /usr/bin/cc
-- Check for working C compiler: /usr/bin/cc -- works
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Detecting C compile features
-- Detecting C compile features - done
-- Check for working CXX compiler: /usr/bin/c++
-- Check for working CXX compiler: /usr/bin/c++ -- works
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Configuring done
-- Generating done
-- Build files have been written to: /mnt/wsl/docker-desktop-bind-mounts/Ubuntu-20.04/fdbd7e23af76ca956070b6eadda658eaefba4c755edccae5481002971e28d70b/course/sp/08-lib/04-cmake/01-hello-cmake/build
wsl> make
make: Warning: File 'Makefile' has modification time 0.0092 s in the future
Scanning dependencies of target hello_cmake
make[2]: Warning: File 'CMakeFiles/hello_cmake.dir/depend.make' has modification time 1.4 s in the future[ 50%] Building CXX object CMakeFiles/hello_cmake.dir/main.cpp.o
[100%] Linking CXX executable hello_cmake
make[2]: warning:  Clock skew detected.  Your build may be incomplete.
[100%] Built target hello_cmake
make: warning:  Clock skew detected.  Your build may be incomplete.
wsl> ./hello_cmake
Hello CMake!
```

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
-- Build files have been written to: C:/ccc/course/sp/01-gcc/06-cmake/01-hello-cmake/build
```

然後用 Visual Studio 開啟 build/hello_cmake.sln 並按下《建置/建置方案》去 build

```
$ cd Debug
$ ls
hello_cmake.exe*  hello_cmake.pdb
$ ./hello_cmake
Hello CMake!
```

## msys2

```
  368  cd A-hello-cmake/
  369  mkdir build && cd build
  370  cmake ..
  371  ninja
  372  ./hello_cmake

Hero3C@DESKTOP-O093POU MINGW64 /c/ccc/course/sa/c/package/cmake
$ cd A-hello-cmake/

Hero3C@DESKTOP-O093POU MINGW64 /c/ccc/course/sa/c/package/cmake/A-hello-cmake
$ mkdir build && cd build

Hero3C@DESKTOP-O093POU MINGW64 /c/ccc/course/sa/c/package/cmake/A-hello-cmake/build
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
-- Build files have been written to: C:/ccc/course/sa/c/package/cmake/A-hello-cmake/build

Hero3C@DESKTOP-O093POU MINGW64 /c/ccc/course/sa/c/package/cmake/A-hello-cmake/build
$ ninja
[1/2] Building CXX object CMakeFiles/hello_cmake.dir/main.cpp.obj
[2/2] Linking CXX executable hello_cmake.exe

Hero3C@DESKTOP-O093POU MINGW64 /c/ccc/course/sa/c/package/cmake/A-hello-cmake/build
$ ./hello_cmake
Hello CMake!
```
