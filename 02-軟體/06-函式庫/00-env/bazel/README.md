# Bazel

* [大型C++工程的构建技术：Modern CMake vs Bazel](https://zhuanlan.zhihu.com/p/112712537)


## 有支援 msys2

https://docs.bazel.build/versions/4.2.1/install-windows.html

epending on which languages you want to build, you will need:

MSYS2 x86_64

MSYS2 is a software distro and building platform for Windows. It contains Bash and common Unix tools (like grep, tar, git).

You will need MSYS2 to build, test, or run targets that depend on Bash. Typically these are genrule, sh_binary, sh_test, but there may be more (e.g. Starlark rules). Bazel shows an error if a build target needs Bash but Bazel could not locate it.

Common MSYS2 packages

You will likely need these to build and run targets that depend on Bash. MSYS2 does not install these tools by default, so you need to install them manually. Projects that depend on Bash tools in PATH need this step (for example TensorFlow).

Open the MSYS2 terminal and run this command:

pacman -S zip unzip patch diffutils git
Optional: If you want to use Bazel from CMD or Powershell and still be able to use Bash tools, make sure to add <MSYS2_INSTALL_PATH>/usr/bin to your PATH environment variable.