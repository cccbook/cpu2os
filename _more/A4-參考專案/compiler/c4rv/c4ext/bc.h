#ifndef __BC_H__
#define __BC_H__

enum { W=4 }; // Word: W = sizeof(int)
enum { SName, SClass, SFields };
enum { Entry, CStart, CSize, COff, DStart, DSize, DOff, TStart, TSize, TOff, SStart, SSize, SOff, RStart, RSize, ROff, HFields };

extern char *bc;
extern int *h, bcLen, ofd;
extern char *symClassNames;
extern int codeLen, stabLen, dataLen;

extern void bc_init();
extern int bc_dump_sym(int len);
extern int bc_save();
extern int bc_load();
extern int bc_dump();
extern void bc_dasm();
extern void bc2x();

#define STACK_SIZE 10000

#define debug(...) fprintf(stderr, __VA_ARGS__)
#define emit(...) printf(__VA_ARGS__);

#endif
