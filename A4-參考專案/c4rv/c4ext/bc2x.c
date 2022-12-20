void bc2x() {
  bc2head();
  pc = code;
  for (int i=0; pc-code<codeLen; i++) {
    bc2fetch();
    int ir = *pc++;
    switch (ir) {
      case IMM:  bc2imm(); break; // a = *pc++;
      case STR:  bc2str(); break; // a = (int)(stab+*pc++);
      case LVAR: bc2lvar(); break; // a = (int)(bp + *pc++);
      case GVAR: bc2gvar(); break; // a = (int)(data+*pc++);
      case JMP:  bc2jmp(); break; // pc = (int *) (pc+*pc);
      case JSR:  bc2jsr(); break; // { *--sp = (int)(pc + 1); pc = (int *)(pc+*pc); }
      case ENT:  bc2ent(); break; // { *--sp = (int)bp; bp = sp; sp = sp - *pc++; }
      case ADJ:  bc2adj(); break; // sp = sp + *pc++;
      case BZ:   bc2bz(); break;// pc = a ? pc + 1 : (int *) (pc+*pc);
      case BNZ:  bc2bnz(); break; // pc = a ? (int *)(pc+*pc) : pc + 1;
      case LEV:  bc2lev(); break; // sp = bp; bp = (int *)*sp++; pc = (int *)*sp++;
      case LI:   bc2li(); break; // load int: a = *(int *)a;  
      case LC:   bc2lc(); break; // load char: a = *(char *)a;
      case SI:   bc2si(); break; // store int *(int *)*sp++ = a;
      case SC:   bc2sc(); break; // store char a = *(char *)*sp++ = a;
      case PSH:  bc2psh(); break; // *--sp = a;
     // 運算指令
      case OR:   bc2or(); break; // a = *sp++ |  a;
      case XOR:  bc2xor(); break; // a = *sp++ ^  a;
      case AND:  bc2and(); break; // a = *sp++ &  a;
      case EQ:   bc2eq(); break; // a = *sp++ == a;
      case NE:   bc2ne(); break; // a = *sp++ != a;
      case LT:   bc2lt(); break; // a = *sp++ <  a;
      case GT:   bc2gt(); break; // a = *sp++ >  a;
      case LE:   bc2le(); break; // a = *sp++ <= a;
      case GE:   bc2ge(); break; // a = *sp++ >= a;
      case SHL:  bc2shl(); break; // a = *sp++ << a;
      case SHR:  bc2shr(); break; // a = *sp++ >> a;
      case ADD:  bc2add(); break; // a = *sp++ +  a;
      case SUB:  bc2sub(); break; // a = *sp++ -  a;
      case MUL:  bc2mul(); break; // a = *sp++ *  a;
      case DIV:  bc2div(); break; // a = *sp++ /  a;
      case MOD:  bc2mod(); break; // a = *sp++ %  a;
      // 系統呼叫
      case OPEN: bc2open(); break; // a = open((char *)sp[2], sp[1], *sp);
      case READ: bc2read(); break; // a = read(sp[2], (char *)sp[1], *sp);
      case WRIT: bc2writ(); break; // a = write(sp[2], (char *)sp[1], *sp);
      case CLOS: bc2clos(); break; // close(*sp);
      case PRTF: bc2prtf(); break; // { t = sp + pc[1]; a = printf((char *)t[-1], t[-2], t[-3], t[-4], t[-5], t[-6]); }
      case SPRT: bc2sprt(); break; // { t = sp + pc[1]; a = sprintf((char *)t[-1], (char *)t[-2], t[-3], t[-4], t[-5], t[-6]); } // sprintf(str,"....", a, b, c, d, e)
      case SSCN: bc2sscn(); break; // { t = sp + pc[1]; a = sscanf((char *)t[-1], (char *)t[-2], t[-3], t[-4], t[-5], t[-6]); } // sprintf(str,"....", a, b, c, d, e)
      case GETS: bc2gets(); break; // gets((char*)*sp);
      case MALC: bc2malc(); break; // a = (int)malloc(*sp);
      case FREE: bc2free(); break; // free((void *)*sp);
      case MSET: bc2mset(); break; // a = (int)memset((char *)sp[2], sp[1], *sp);
      case MCMP: bc2mcmp(); break;  // a = memcmp((char *)sp[2], (char *)sp[1], *sp);
      case MCPY: bc2mcpy(); break; // a = (int)memcpy((char *)sp[2], (char *)sp[1], *sp);
      case EXIT: bc2exit(); break; // return
      default: emit("Error: unknown command %d", ir);
    }
    emit("\n");
  }
  bc2tail();
}

int main(int argc, char **argv) { // 主程式
  oFile = argv[1];
  init();
  bc_load();
  bc2x();
}
