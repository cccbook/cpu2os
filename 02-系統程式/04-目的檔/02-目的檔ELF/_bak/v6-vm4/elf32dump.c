#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <elf.h>
#include <stdbool.h>

#include "lib.c"
#include "dasm.c"
#include "elf32lib.c"

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        printf("使用方式: %s <ELF 檔案>\n", argv[0]);
        return 1;
    }

    FILE *file = fopen(argv[1], "rb");
    if (!file)
    {
        perror("無法開啟檔案");
        return 1;
    }

    Elf32_Ehdr elf_header;
    fread(&elf_header, 1, sizeof(Elf32_Ehdr), file);

    dump_elf_header(file, elf_header);
    dump_elf_sections(file, elf_header);

    fclose(file);
    return 0;
}
