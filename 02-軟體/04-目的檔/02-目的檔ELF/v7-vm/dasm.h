#pragma once

#include <stdint.h>

char disassemble(uint32_t instr, char *line);
void disassemble_block(char *block, int size);

