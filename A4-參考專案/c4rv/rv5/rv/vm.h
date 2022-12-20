#ifndef __VM_H__
#define __VM_H__

#include "obj.h"

extern int isDebug;
extern void exec();
extern void vm(uint8_t *m, uint32_t mTop);

#endif
