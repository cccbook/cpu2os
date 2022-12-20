#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "c4.h"

#define MMAX 10000
static int code[]={8,2,1,-1,15,0,0,13,1,-2,15,0,1,13,3,0,15,1,-2,11,15,1,-1,11,15,36,9,3,1,-2,11,15,1,2,11,23,6,42,3,12,15,1,-2,11,15,1,-1,11,15,36,9,3,1,-1,15,1,-1,11,15,1,-2,11,27,13,1,-2,15,11,15,0,1,27,13,15,0,1,28,4,-50,1,-1,11,10,10,8,0,3,24,15,36,9,1,3,40,15,0,10,15,5,-99,9,1,15,36,9,2,10};
static char stab[]={105,61,37,100,32,115,61,37,100,10,0,0,105,61,37,100,32,115,61,37,100,10,0,0,109,97,105,110,40,41,32,115,116,97,114,116,10,0,0,0,115,117,109,40,49,48,41,61,37,100,10,0};
static char data[]={};
static int stack[MMAX];

int *sp, *bp, *pc; 

int vm_run(int *pc) {

static const int _pc[]={&&pc0-&&pc0,&&pc1-&&pc0,&&pc2-&&pc0,&&pc3-&&pc0,&&pc4-&&pc0,&&pc5-&&pc0,&&pc6-&&pc0,&&pc7-&&pc0,&&pc8-&&pc0,&&pc9-&&pc0,&&pc10-&&pc0,&&pc11-&&pc0,&&pc12-&&pc0,&&pc13-&&pc0,&&pc14-&&pc0,&&pc15-&&pc0,&&pc16-&&pc0,&&pc17-&&pc0,&&pc18-&&pc0,&&pc19-&&pc0,&&pc20-&&pc0,&&pc21-&&pc0,&&pc22-&&pc0,&&pc23-&&pc0,&&pc24-&&pc0,&&pc25-&&pc0,&&pc26-&&pc0,&&pc27-&&pc0,&&pc28-&&pc0,&&pc29-&&pc0,&&pc30-&&pc0,&&pc31-&&pc0,&&pc32-&&pc0,&&pc33-&&pc0,&&pc34-&&pc0,&&pc35-&&pc0,&&pc36-&&pc0,&&pc37-&&pc0,&&pc38-&&pc0,&&pc39-&&pc0,&&pc40-&&pc0,&&pc41-&&pc0,&&pc42-&&pc0,&&pc43-&&pc0,&&pc44-&&pc0,&&pc45-&&pc0,&&pc46-&&pc0,&&pc47-&&pc0,&&pc48-&&pc0,&&pc49-&&pc0,&&pc50-&&pc0,&&pc51-&&pc0,&&pc52-&&pc0,&&pc53-&&pc0,&&pc54-&&pc0,&&pc55-&&pc0,&&pc56-&&pc0,&&pc57-&&pc0,&&pc58-&&pc0,&&pc59-&&pc0,&&pc60-&&pc0,&&pc61-&&pc0,&&pc62-&&pc0,&&pc63-&&pc0,&&pc64-&&pc0,&&pc65-&&pc0,&&pc66-&&pc0,&&pc67-&&pc0,&&pc68-&&pc0,&&pc69-&&pc0,&&pc70-&&pc0,&&pc71-&&pc0,&&pc72-&&pc0,&&pc73-&&pc0,&&pc74-&&pc0,&&pc75-&&pc0,&&pc76-&&pc0,&&pc77-&&pc0,&&pc78-&&pc0,&&pc79-&&pc0,&&pc80-&&pc0,&&pc81-&&pc0,&&pc82-&&pc0,&&pc83-&&pc0,&&pc84-&&pc0,&&pc85-&&pc0,&&pc86-&&pc0,&&pc87-&&pc0,&&pc88-&&pc0,&&pc89-&&pc0,&&pc90-&&pc0,&&pc91-&&pc0,&&pc92-&&pc0,&&pc93-&&pc0,&&pc94-&&pc0,&&pc95-&&pc0,&&pc96-&&pc0,&&pc97-&&pc0,&&pc98-&&pc0,&&pc99-&&pc0,&&pc100-&&pc0,&&pc101-&&pc0,&&pc102-&&pc0,&&pc103-&&pc0,&&pc104-&&pc0,&&pc105-&&pc0,&&pc106-&&pc0};
int levCount = 0;
int a, *t;
goto *(&&pc0+_pc[pc-code]);pc0:pc++; /*ENT 2*/ pc1:*--sp = (int)bp; bp = sp; sp = sp - *pc++;
pc2:pc++; /*LVAR -1*/ pc3:a = (int)(bp + *pc++);
pc4:pc++; /*PSH*/ *--sp = a;
pc5:pc++; /*IMM 0*/ pc6:a = *pc++;
pc7:pc++; /*SI*/ *(int *)*sp++ = a;
pc8:pc++; /*LVAR -2*/ pc9:a = (int)(bp + *pc++);
pc10:pc++; /*PSH*/ *--sp = a;
pc11:pc++; /*IMM 1*/ pc12:a = *pc++;
pc13:pc++; /*SI*/ *(int *)*sp++ = a;
pc14:pc++; /*STR 0*/ pc15:a = (int)(stab+*pc++);
pc16:pc++; /*PSH*/ *--sp = a;
pc17:pc++; /*LVAR -2*/ pc18:a = (int)(bp + *pc++);
pc19:pc++; /*LI*/ a = *(int *)a;
pc20:pc++; /*PSH*/ *--sp = a;
pc21:pc++; /*LVAR -1*/ pc22:a = (int)(bp + *pc++);
pc23:pc++; /*LI*/ a = *(int *)a;
pc24:pc++; /*PSH*/ *--sp = a;
pc25:pc++; /*PRTF*/ t = sp + pc[1]; a = printf((char *)t[-1], t[-2], t[-3], t[-4], t[-5], t[-6]);
pc26:pc++; /*ADJ 3*/ pc27:sp = sp + *pc++;
pc28:pc++; /*LVAR -2*/ pc29:a = (int)(bp + *pc++);
pc30:pc++; /*LI*/ a = *(int *)a;
pc31:pc++; /*PSH*/ *--sp = a;
pc32:pc++; /*LVAR 2*/ pc33:a = (int)(bp + *pc++);
pc34:pc++; /*LI*/ a = *(int *)a;
pc35:pc++; /*LE*/ a = *sp++ <= a;
pc36:pc++; /*BZ*/ if (!a) pc=(int *) (pc+*pc); goto *(&&pc0+_pc[pc-code]);;
pc37:pc++; /*MSET*/ a = (int)memset((char *)sp[2], sp[1], *sp);
pc38:pc++; /*STR 12*/ pc39:a = (int)(stab+*pc++);
pc40:pc++; /*PSH*/ *--sp = a;
pc41:pc++; /*LVAR -2*/ pc42:a = (int)(bp + *pc++);
pc43:pc++; /*LI*/ a = *(int *)a;
pc44:pc++; /*PSH*/ *--sp = a;
pc45:pc++; /*LVAR -1*/ pc46:a = (int)(bp + *pc++);
pc47:pc++; /*LI*/ a = *(int *)a;
pc48:pc++; /*PSH*/ *--sp = a;
pc49:pc++; /*PRTF*/ t = sp + pc[1]; a = printf((char *)t[-1], t[-2], t[-3], t[-4], t[-5], t[-6]);
pc50:pc++; /*ADJ 3*/ pc51:sp = sp + *pc++;
pc52:pc++; /*LVAR -1*/ pc53:a = (int)(bp + *pc++);
pc54:pc++; /*PSH*/ *--sp = a;
pc55:pc++; /*LVAR -1*/ pc56:a = (int)(bp + *pc++);
pc57:pc++; /*LI*/ a = *(int *)a;
pc58:pc++; /*PSH*/ *--sp = a;
pc59:pc++; /*LVAR -2*/ pc60:a = (int)(bp + *pc++);
pc61:pc++; /*LI*/ a = *(int *)a;
pc62:pc++; /*ADD*/ a = *sp++ +  a;
pc63:pc++; /*SI*/ *(int *)*sp++ = a;
pc64:pc++; /*LVAR -2*/ pc65:a = (int)(bp + *pc++);
pc66:pc++; /*PSH*/ *--sp = a;
pc67:pc++; /*LI*/ a = *(int *)a;
pc68:pc++; /*PSH*/ *--sp = a;
pc69:pc++; /*IMM 1*/ pc70:a = *pc++;
pc71:pc++; /*ADD*/ a = *sp++ +  a;
pc72:pc++; /*SI*/ *(int *)*sp++ = a;
pc73:pc++; /*PSH*/ *--sp = a;
pc74:pc++; /*IMM 1*/ pc75:a = *pc++;
pc76:pc++; /*SUB*/ a = *sp++ -  a;
pc77:pc++; /*JMP -50*/ pc78:pc = (int *) (pc+*pc); goto *(&&pc0+_pc[pc-code]);
pc79:pc++; /*LVAR -1*/ pc80:a = (int)(bp + *pc++);
pc81:pc++; /*LI*/ a = *(int *)a;
pc82:pc++; /*LEV*/ levCount++; sp = bp; bp = (int *)*sp++; pc = (int *)*sp++;if (pc==NULL) return 0; else goto *(&&pc0+_pc[pc-code]);
pc83:pc++; /*LEV*/ levCount++; sp = bp; bp = (int *)*sp++; pc = (int *)*sp++;if (pc==NULL) return 0; else goto *(&&pc0+_pc[pc-code]);
pc84:pc++; /*ENT 0*/ pc85:*--sp = (int)bp; bp = sp; sp = sp - *pc++;
pc86:pc++; /*STR 24*/ pc87:a = (int)(stab+*pc++);
pc88:pc++; /*PSH*/ *--sp = a;
pc89:pc++; /*PRTF*/ t = sp + pc[1]; a = printf((char *)t[-1], t[-2], t[-3], t[-4], t[-5], t[-6]);
pc90:pc++; /*ADJ 1*/ pc91:sp = sp + *pc++;
pc92:pc++; /*STR 40*/ pc93:a = (int)(stab+*pc++);
pc94:pc++; /*PSH*/ *--sp = a;
pc95:pc++; /*IMM 10*/ pc96:a = *pc++;
pc97:pc++; /*PSH*/ *--sp = a;
pc98:pc++; /*JSR -99*/ pc99:*--sp = (int)(pc + 1); pc = (int *)(pc+*pc); goto *(&&pc0+_pc[pc-code]);
pc100:pc++; /*ADJ 1*/ pc101:sp = sp + *pc++;
pc102:pc++; /*PSH*/ *--sp = a;
pc103:pc++; /*PRTF*/ t = sp + pc[1]; a = printf((char *)t[-1], t[-2], t[-3], t[-4], t[-5], t[-6]);
pc104:pc++; /*ADJ 2*/ pc105:sp = sp + *pc++;
printf("before LEV\n");
pc106:pc++; /*LEV*/ levCount++; sp = bp; bp = (int *)*sp++; pc = (int *)*sp++;
printf("pc=%p code=%p pc-code=%d\n", pc, code, pc-code);
printf("sp=%p stack=%p sp-(stack+10000)=%d\n", sp, stack, sp-(stack+10000-1));
printf("levCount=%d\n", levCount);
if (pc==NULL) return 0; else goto *(&&pc0+_pc[pc-code]);
}

int vm_main(int *pc, int argc, char **argv) {
  int *bp, *sp, *t;
  // setup stack
  bp = sp = (int *)((int)stack + MMAX); // 將 sp, bp 指向堆疊起始點 (底端)
  *--sp = EXIT; // call exit if main returns，執行到此 EXIT 指令就會離開 vm_run()
  *--sp = PSH;  // ??
  t = sp;       // ??
  *--sp = argc; // argc, argv 是要傳給 main 的參數。
  *--sp = (int)argv;
  *--sp = (int)t;  // ??
  vm_run(pc, bp, sp, data, stab); // 開始執行 pc 指向的 main(argc,argv) 函數
  return 0;
}

int main(int argc, char *argv[]) {
  *pc=code+84; 
  vm_main(pc, argc, argv, data, stab);
}
