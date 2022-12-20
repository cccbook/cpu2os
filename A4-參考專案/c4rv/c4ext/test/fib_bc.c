#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
int run() {

static int code[]={8,0,1,2,11,15,0,0,23,6,4,0,0,10,1,2,11,15,0,1,19,6,4,0,1,10,1,2,11,15,0,1,28,15,5,-35,9,1,15,1,2,11,15,0,2,28,15,5,-48,9,1,27,10,10,8,0,3,0,15,0,7,15,5,-63,9,1,15,36,9,2,10};
static char stab[]={102,40,55,41,61,37,100,10,0,0,0,0};
static char data[]={};
static int stack[10000];
static const void* _pc[]={&&pc0,&&pc1,&&pc2,&&pc3,&&pc4,&&pc5,&&pc6,&&pc7,&&pc8,&&pc9,&&pc10,&&pc11,&&pc12,&&pc13,&&pc14,&&pc15,&&pc16,&&pc17,&&pc18,&&pc19,&&pc20,&&pc21,&&pc22,&&pc23,&&pc24,&&pc25,&&pc26,&&pc27,&&pc28,&&pc29,&&pc30,&&pc31,&&pc32,&&pc33,&&pc34,&&pc35,&&pc36,&&pc37,&&pc38,&&pc39,&&pc40,&&pc41,&&pc42,&&pc43,&&pc44,&&pc45,&&pc46,&&pc47,&&pc48,&&pc49,&&pc50,&&pc51,&&pc52,&&pc53,&&pc54,&&pc55,&&pc56,&&pc57,&&pc58,&&pc59,&&pc60,&&pc61,&&pc62,&&pc63,&&pc64,&&pc65,&&pc66,&&pc67,&&pc68,&&pc69,&&pc70};
int *pc=code+54, *sp = stack+10000, *bp=sp, a, *t;
goto *_pc[pc-code];
pc0:pc++; /*ENT 0*/ pc1:*--sp = (int)bp; bp = sp; sp = sp - *pc++;
pc2:pc++; /*LVAR 2*/ pc3:a = (int)(bp + *pc++);
pc4:pc++; /*LI*/ a = *(int *)a;
pc5:pc++; /*PSH*/ *--sp = a;
pc6:pc++; /*IMM 0*/ pc7:a = *pc++;
pc8:pc++; /*LE*/ a = *sp++ <= a;
pc9:pc++; /*BZ 4*/ pc10:if (!a) { pc=(int *)(pc+*pc); goto *_pc[pc-code];} else pc++;
pc11:pc++; /*IMM 0*/ pc12:a = *pc++;
pc13:pc++; /*LEV*/ sp = bp; bp = (int *)*sp++; pc = (int *)*sp++;if (pc==NULL) return 0; else goto *_pc[pc-code];
pc14:pc++; /*LVAR 2*/ pc15:a = (int)(bp + *pc++);
pc16:pc++; /*LI*/ a = *(int *)a;
pc17:pc++; /*PSH*/ *--sp = a;
pc18:pc++; /*IMM 1*/ pc19:a = *pc++;
pc20:pc++; /*EQ*/ a = *sp++ == a;
pc21:pc++; /*BZ 4*/ pc22:if (!a) { pc=(int *)(pc+*pc); goto *_pc[pc-code];} else pc++;
pc23:pc++; /*IMM 1*/ pc24:a = *pc++;
pc25:pc++; /*LEV*/ sp = bp; bp = (int *)*sp++; pc = (int *)*sp++;if (pc==NULL) return 0; else goto *_pc[pc-code];
pc26:pc++; /*LVAR 2*/ pc27:a = (int)(bp + *pc++);
pc28:pc++; /*LI*/ a = *(int *)a;
pc29:pc++; /*PSH*/ *--sp = a;
pc30:pc++; /*IMM 1*/ pc31:a = *pc++;
pc32:pc++; /*SUB*/ a = *sp++ -  a;
pc33:pc++; /*PSH*/ *--sp = a;
pc34:pc++; /*JSR -35*/ pc35:*--sp = (int)(pc + 1); pc = (int *)(pc+*pc); goto *_pc[pc-code];
pc36:pc++; /*ADJ 1*/ pc37:sp = sp + *pc++;
pc38:pc++; /*PSH*/ *--sp = a;
pc39:pc++; /*LVAR 2*/ pc40:a = (int)(bp + *pc++);
pc41:pc++; /*LI*/ a = *(int *)a;
pc42:pc++; /*PSH*/ *--sp = a;
pc43:pc++; /*IMM 2*/ pc44:a = *pc++;
pc45:pc++; /*SUB*/ a = *sp++ -  a;
pc46:pc++; /*PSH*/ *--sp = a;
pc47:pc++; /*JSR -48*/ pc48:*--sp = (int)(pc + 1); pc = (int *)(pc+*pc); goto *_pc[pc-code];
pc49:pc++; /*ADJ 1*/ pc50:sp = sp + *pc++;
pc51:pc++; /*ADD*/ a = *sp++ +  a;
pc52:pc++; /*LEV*/ sp = bp; bp = (int *)*sp++; pc = (int *)*sp++;if (pc==NULL) return 0; else goto *_pc[pc-code];
pc53:pc++; /*LEV*/ sp = bp; bp = (int *)*sp++; pc = (int *)*sp++;if (pc==NULL) return 0; else goto *_pc[pc-code];
pc54:pc++; /*ENT 0*/ pc55:*--sp = (int)bp; bp = sp; sp = sp - *pc++;
pc56:pc++; /*STR 0*/ pc57:a = (int)(stab+*pc++);
pc58:pc++; /*PSH*/ *--sp = a;
pc59:pc++; /*IMM 7*/ pc60:a = *pc++;
pc61:pc++; /*PSH*/ *--sp = a;
pc62:pc++; /*JSR -63*/ pc63:*--sp = (int)(pc + 1); pc = (int *)(pc+*pc); goto *_pc[pc-code];
pc64:pc++; /*ADJ 1*/ pc65:sp = sp + *pc++;
pc66:pc++; /*PSH*/ *--sp = a;
pc67:pc++; /*PRTF*/ t = sp + pc[1]; a = printf((char *)t[-1], t[-2], t[-3], t[-4], t[-5], t[-6]);
pc68:pc++; /*ADJ 2*/ pc69:sp = sp + *pc++;
pc70:pc++; /*LEV*/ sp = bp; bp = (int *)*sp++; pc = (int *)*sp++;if (pc==NULL) return 0; else goto *_pc[pc-code];
}
int main() {
  run();
}
