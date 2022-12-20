# Linux 套件安裝

## Ubuntu

> apt install `<package>`

apt 常用命令

```
列出所有可更新的软件清单命令：sudo apt update

升级软件包：sudo apt upgrade

列出可更新的软件包及版本信息：apt list --upgradeable

升级软件包，升级前先删除需要更新软件包：sudo apt full-upgrade

安装指定的软件命令：sudo apt install <package_name>

安装多个软件包：sudo apt install <package_1> <package_2> <package_3>

更新指定的软件命令：sudo apt update <package_name>

显示软件包具体信息,例如：版本号，安装大小，依赖关系等等：sudo apt show <package_name>

删除软件包命令：sudo apt remove <package_name>

清理不再使用的依赖和库文件: sudo apt autoremove

移除软件包及配置文件: sudo apt purge <package_name>

查找软件包命令： sudo apt search <keyword>

列出所有已安装的包：apt list --installed

列出所有已安装的包的版本信息：apt list --all-versions
```

## ArchLinux / Msys2

```
pacman -S `<package>`

pacman -Ss `keyword`
```

