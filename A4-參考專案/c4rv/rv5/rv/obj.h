#ifndef __OBJ_H__
#define __OBJ_H__

#include "cpu.h"

extern int objDump(char *objFile);
extern void ir2asm(uint32_t ir, char *line);

#endif
