# qemu-user-static

* [RISC-V simulation with Qemu](https://medium.com/@e1d1/risc-v-simulation-with-qemu-61ea8f2d8f4b)

```
echo -e '#include <stdio.h>\nint main(){\n printf("hi\\n");\n return 0;\n}' > hi.c
riscv64-unknown-elf-gcc -o hi hi.c 
qemu-riscv64-static hi
```

## 