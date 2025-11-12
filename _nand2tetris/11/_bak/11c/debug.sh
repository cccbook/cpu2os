gcc -g -O0 -o jack2vm jack2vm.c
# lldb --args ./jack2vm Square/Main.jack Square/Main.vm
lldb -- ./jack2vm Square/Main.jack Square/Main.vm
# lldb -- ./jack2vm main.jack output.vm