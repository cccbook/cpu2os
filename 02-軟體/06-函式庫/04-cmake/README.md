# cmake

* https://github.com/ttroy50/cmake-examples (超讚)

# A-hello-cmake

```
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