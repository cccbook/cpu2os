安裝成功，執行 CMake 失敗！


Hero3C@DESKTOP-O093POU MINGW64 ~
$ git clone https://github.com/conan-io/conan.git conan-io
Cloning into 'conan-io'...
remote: Enumerating objects: 60511, done.
remote: Counting objects: 100% (9147/9147), done.
remote: Compressing objects: 100% (4135/4135), done.
remote: Total 60511 (delta 6512), reused 6753 (delta 4988), pack-reused 51364
Receiving objects: 100% (60511/60511), 19.38 MiB | 562.00 KiB/s, done.
Resolving deltas: 100% (48433/48433), done.
Updating files: 100% (1159/1159), done.

Hero3C@DESKTOP-O093POU MINGW64 ~
$ cd conan-io

Hero3C@DESKTOP-O093POU MINGW64 ~/conan-io
$ pip install -e .
Obtaining file:///home/Hero3C/conan-io
Collecting PyJWT<2.0.0,>=1.4.0
  Downloading PyJWT-1.7.1-py2.py3-none-any.whl (18 kB)
Collecting requests<3.0.0,>=2.25
  Downloading requests-2.26.0-py2.py3-none-any.whl (62 kB)
     |████████████████████████████████| 62 kB 180 kB/s
Collecting urllib3<1.27,>=1.26.6
  Downloading urllib3-1.26.7-py2.py3-none-any.whl (138 kB)
     |████████████████████████████████| 138 kB 458 kB/s
Collecting colorama<0.5.0,>=0.3.3
  Downloading colorama-0.4.4-py2.py3-none-any.whl (16 kB)
Collecting PyYAML<6.0,>=3.11
  Downloading PyYAML-5.4.1.tar.gz (175 kB)
     |████████████████████████████████| 175 kB 607 kB/s
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
    Preparing wheel metadata ... done
Collecting patch-ng<1.18,>=1.17.4
  Downloading patch-ng-1.17.4.tar.gz (17 kB)
Collecting fasteners>=0.14.1
  Downloading fasteners-0.16.3-py2.py3-none-any.whl (28 kB)
Collecting six<=1.16.0,>=1.10.0
  Downloading six-1.16.0-py2.py3-none-any.whl (11 kB)
Collecting node-semver==0.6.1
  Downloading node_semver-0.6.1-py3-none-any.whl (10 kB)
Collecting distro<=1.6.0,>=1.0.2
  Downloading distro-1.6.0-py2.py3-none-any.whl (19 kB)
Collecting pygments<3.0,>=2.0
  Downloading Pygments-2.10.0-py3-none-any.whl (1.0 MB)
     |████████████████████████████████| 1.0 MB 695 kB/s
Collecting tqdm<5,>=4.28.1
  Downloading tqdm-4.62.3-py2.py3-none-any.whl (76 kB)
     |████████████████████████████████| 76 kB 539 kB/s
Collecting Jinja2<3,>=2.9
  Downloading Jinja2-2.11.3-py2.py3-none-any.whl (125 kB)
     |████████████████████████████████| 125 kB 617 kB/s
Collecting python-dateutil<3,>=2.7.0
  Downloading python_dateutil-2.8.2-py2.py3-none-any.whl (247 kB)
     |████████████████████████████████| 247 kB 637 kB/s
Collecting bottle<0.13,>=0.12.8
  Downloading bottle-0.12.19-py3-none-any.whl (89 kB)
     |████████████████████████████████| 89 kB 484 kB/s
Collecting pluginbase>=0.5
  Downloading pluginbase-1.0.1.tar.gz (43 kB)
     |████████████████████████████████| 43 kB 386 kB/s
Collecting MarkupSafe>=0.23
  Downloading MarkupSafe-2.0.1.tar.gz (18 kB)
Collecting charset-normalizer~=2.0.0
  Downloading charset_normalizer-2.0.8-py3-none-any.whl (39 kB)
Collecting certifi>=2017.4.17
  Downloading certifi-2021.10.8-py2.py3-none-any.whl (149 kB)
     |████████████████████████████████| 149 kB 632 kB/s
Collecting idna<4,>=2.5
  Downloading idna-3.3-py3-none-any.whl (61 kB)
     |████████████████████████████████| 61 kB 610 kB/s
Using legacy 'setup.py install' for MarkupSafe, since package 'wheel' is not installed.
Using legacy 'setup.py install' for patch-ng, since package 'wheel' is not installed.
Using legacy 'setup.py install' for pluginbase, since package 'wheel' is not installed.
Building wheels for collected packages: PyYAML
  Building wheel for PyYAML (PEP 517) ... done
  Created wheel for PyYAML: filename=PyYAML-5.4.1-cp39-cp39-mingw64_nt_10_0_19043_3_2_0_340_x86_64_x86_64.whl size=45680 sha256=f9094f72039147b21e5cc19f7923738429ac759d3091b317af70c5e19469ccb9
  Stored in directory: /home/Hero3C/.cache/pip/wheels/b7/a5/c4/504d913c2a55bb09c607541578ec5f844d1ff33467abe93ba5
Successfully built PyYAML
Installing collected packages: urllib3, six, MarkupSafe, idna, charset-normalizer, certifi, tqdm, requests, PyYAML, python-dateutil, PyJWT, pygments, pluginbase, patch-ng, node-semver, Jinja2, fasteners, distro, colorama, bottle, conan
    Running setup.py install for MarkupSafe ... done
    Running setup.py install for pluginbase ... done
    Running setup.py install for patch-ng ... done
  Running setup.py develop for conan
Successfully installed Jinja2-2.11.3 MarkupSafe-2.0.1 PyJWT-1.7.1 PyYAML-5.4.1 bottle-0.12.19 certifi-2021.10.8 charset-normalizer-2.0.8 colorama-0.4.4 conan-1.43.0.dev0 distro-1.6.0 fasteners-0.16.3 idna-3.3 node-semver-0.6.1 patch-ng-1.17.4 pluginbase-1.0.1 pygments-2.10.0 python-dateutil-2.8.2 requests-2.26.0 six-1.16.0 tqdm-4.62.3 urllib3-1.26.7

Hero3C@DESKTOP-O093POU MINGW64 /c/ccc/course/sa/c/package/conan/md5/build
$ cmake .. -G Ninja
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
-- Conan: Adjusting output directories
-- Conan: Using cmake global configuration
-- Conan: Adjusting default RPATHs Conan policies
-- Conan: Adjusting language standard
-- Current conanbuildinfo.cmake directory: C:/ccc/course/sa/c/package/conan/md5/build
CMake Error at build/conanbuildinfo.cmake:1365 (message):
  Incorrect 'Visual Studio'.  Toolset specifies compiler as 'MSVC' but CMake
  detected 'GNU'
Call Stack (most recent call first):
  build/conanbuildinfo.cmake:934 (conan_check_compiler)
  CMakeLists.txt:19 (conan_basic_setup)


-- Configuring incomplete, errors occurred!
See also "C:/ccc/course/sa/c/package/conan/md5/build/CMakeFiles/CMakeOutput.log".

