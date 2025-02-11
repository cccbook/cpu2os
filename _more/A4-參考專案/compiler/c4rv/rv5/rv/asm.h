#ifndef __ASM_H__
#define __ASM_H__

#include "../lib/map.h"
#include "cpu.h"
#include "parser.h"

#define SYM_MAX 9999
extern uint32_t asm2ir(char *line);
extern void assemble(char *file);

#endif
