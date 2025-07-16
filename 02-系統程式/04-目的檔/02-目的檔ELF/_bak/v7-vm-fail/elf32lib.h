#pragma once

#include <stdio.h>
#include <elf.h>

void dump_elf_header(FILE *file, Elf32_Ehdr elf_header);
void dump_elf_sections(FILE *file, Elf32_Ehdr elf_header);
