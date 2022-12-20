#include "c4.h"

int *h, bcLen, ofd;
int codeLen, stabLen, dataLen;

void stab_dump(char *stab, int len) {
  for (int i=0; i<len; i++) {
    if (stab[i] == 0) printf(" ");
    else if (stab[i] == '\n') printf("\\n");
    else printf("%c", stab[i]);
  }
  printf("\n");
}

int bc_dump_header() { // 印出目的檔擋頭。
  printf("header: entry=%d\n\n", h[Entry]);
  printf("          Size      VMA      LMA   Offset  \n");
  printf("code: %08X %08X %8p %08X\n", h[CSize], h[CStart], code, h[COff]); 
  printf("data: %08X %08X %8p %08X\n", h[DSize], h[DStart], data, h[DOff]);
  printf("relo: %08X %08X %8p %08X\n", h[RSize], h[RStart], rel,  h[ROff]);
  printf("stab: %08X %08X %8p %08X\n", h[TSize], h[TStart], stab, h[TOff]);
  printf("syms: %08X %08X %8p %08X\n", h[SSize], h[SStart], symstab, h[SOff]);
  return 0;
}

int bc_save() { // 儲存目的檔
  char *o;
  h = (int*) bc;
  h[Entry] = pc-code;
  o = bc + HFields * W;
  h[CStart]=(int)code; h[COff] = o-bc; h[CSize] = (int)(e+1-code)*W; memcpy(o, code, h[CSize]); o = o + h[CSize];
  h[DStart]=(int)data; h[DOff] = o-bc; h[DSize] = (int)(datap-data); memcpy(o, data, h[DSize]); o = o + h[DSize];
  h[RStart]=(int)rel;  h[ROff] = o-bc; h[RSize] = (int)(relp-rel)*W; memcpy(o, rel,  h[RSize]); o = o + h[RSize];
  h[TStart]=(int)stab; h[TOff] = o-bc; h[TSize] = (int)(stabp-stab); memcpy(o, stab, h[TSize]); o = o + h[TSize];
  h[SStart]=(int)symstab; h[SOff] = o-bc; h[SSize] = (int)(symstabp-symstab); memcpy(o, symstab, h[SSize]); o = o + h[SSize];
  printf("bc_save: symstab=%p symstabp=%p size=%d\n", symstab, symstabp, symstabp-symstab);
  // h[SStart]=(int)sym;  h[SOff] = o-bc; h[SSize] = bc_add_sym((int*)o);                               o = o + h[SSize];
  bcLen = o-bc;
  printf("---------bc_save()-------------\n");
  bc_dump_header();
  if ((ofd = open(oFile, O_WRITE, 0644)) < 0) { printf("could not open(%s)\n", oFile); return -1; }
  write(ofd, bc, bcLen);
  close(ofd);
  return 0;
}

int bc_read() { // 讀取目的檔
  if ((ofd = open(oFile, O_READ, 0644)) < 0) { printf("could not open(%s)\n", oFile); return -1; }
  bcLen = read(ofd, bc, poolsz); // 這裡如果用 Mingw32 的 read() 會有讀到《結束字元》就停止的問題，因此改用 c6.h 裡的 fread(..."rb") 定義。
  close(ofd);
  return bcLen;
}

int bc_head() { // 取得目的檔擋頭，設定對應目的檔變數。
  h = (int*) bc;
  code0= (int*)h[CStart];
  data0= (char*)h[DStart];
  code = (int*)(bc+h[COff]);
  data = (char*)(bc+h[DOff]);
  rel  = (int*)(bc+h[ROff]);
  stab = (char*)(bc+h[TOff]);
  symstab = (char*)(bc+h[SOff]);
  pc   = code+h[Entry];
  codeLen = h[CSize]/W;
  stabLen = h[TSize];
  dataLen = h[DSize];
  return 0;
}

int bc_relocate() { // 根據 rel 紀錄進行重定位。
  int len, *cp, *relp;
  len = h[RSize]/W;
  relp = rel;
  while (relp - rel < len) {
    cp = code + *relp++; // offset
    if (*relp++ == RCode) { // type
      *cp = (int) (((int*)(*cp)-code0) + code);
    } else {
      *cp = (int) (((char*)(*cp)-data0) + data);
    }
  }
  return 0;
}

int bc_load() { // 載入目的檔
  bc_read(); 
  bc_head();
  bc_relocate();
  return 0;
}

int bc_dump() { // 傾印目的檔
  printf("---------bc_dump()-------------\n");
  bc_dump_header();
  printf("string table:"); stab_dump(stab, h[TSize]);
  printf("symbol string table:"); stab_dump(symstab, h[SSize]);
  // int len = h[SSize]/W;
  // bc_dump_sym(len);
  return 0;
}

void bc_dasm() {
  int codeLen = h[CSize]/W;
  int *pc = code;
  for (int i=0; pc-code<codeLen; i++) {
    int ir = *pc++;
    printf("%04X:%08X ", i, ir);
    switch (ir) {
      case IMM: printf("IMM %d", *pc++); break;
      case STR: printf("STR %d", *pc++); break;
      case LVAR: printf("LVAR %d", *pc++); break;
      case GVAR: printf("GVAR %d", *pc++); break;
      case JMP: printf("JMP %d", *pc++); break;
      case JSR: printf("JSR %d", *pc++); break;
      case BZ: printf("BZ %d", *pc++); break;
      case BNZ: printf("BNZ %d", *pc++); break;
      case ENT: printf("ENT %d", *pc++); break;
      case ADJ: printf("ADJ %d", *pc++); break;
      case LEV: printf("LEV"); break;
      case LI: printf("LI"); break;
      case LC: printf("LC"); break;
      case SI: printf("SI"); break;
      case SC: printf("SC"); break;
      case PSH: printf("PSH"); break;
      case OR: printf("OR"); break;
      case XOR: printf("XOR"); break;
      case AND: printf("AND"); break;
      case EQ: printf("EQ"); break;
      case NE: printf("NE"); break;
      case LT: printf("LT"); break;
      case GT: printf("GT"); break;
      case LE: printf("LE"); break;
      case GE: printf("GE"); break;
      case SHL: printf("SHL"); break;
      case SHR: printf("SHR"); break;
      case ADD: printf("ADD"); break;
      case SUB: printf("SUB"); break;
      case MUL: printf("MUL"); break;
      case DIV: printf("DIV"); break;
      case MOD: printf("MOD"); break;
      case OPEN: printf("OPEN"); break;
      case READ: printf("READ"); break;
      case WRIT: printf("WRIT"); break;
      case CLOS: printf("CLOS"); break;
      case PRTF: printf("PRTF"); break;
      case SPRT: printf("SPRT"); break;
      case SSCN: printf("SSCN"); break;
      case GETS: printf("GETS"); break;
      case MALC: printf("MALC"); break;
      case FREE: printf("FREE"); break;
      case MSET: printf("MSET"); break;
      case MCMP: printf("MCMP"); break;
      case MCPY: printf("MCPY"); break;
      case EXIT: printf("EXIT"); break;
      default: printf("Error: unknown command %d\n", ir);
    }
    printf("\n");
  }
}
