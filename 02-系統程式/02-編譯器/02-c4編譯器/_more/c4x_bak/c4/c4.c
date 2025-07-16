// c4.c - C in four functions

// char, int, and pointer types
// if, while, return, and expression statements
// just enough features to allow self-compilation and a bit more

// Written by Robert Swierczek
// 修改者: 陳鍾誠 (模組化並加上中文註解)

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdint.h>
#define int int64_t // 64 bit 電腦

int debug = 1, src=0;

// opcodes (機器碼的 op)
enum { LLA ,IMM ,JMP ,JSR ,BZ  ,BNZ ,ENT ,ADJ ,LEV ,LI  ,LC  ,SI  ,SC  ,PSH ,
       OR  ,XOR ,AND ,EQ  ,NE  ,LT  ,GT  ,LE  ,GE  ,SHL ,SHR ,ADD ,SUB ,MUL ,DIV ,MOD ,
       OPEN,READ,CLOS,PRTF,MALC,FREE,MSET,MCMP,EXIT };

int run(int *pc, int *bp, int *sp) { // 虛擬機 => pc: 程式計數器, sp: 堆疊暫存器, bp: 框架暫存器
  int a, cycle; // a: 累積器, cycle: 執行指令數
  int i, *t;    // temps

  cycle = 0;
  while (1) {
    i = *pc++; ++cycle;
    if (debug) {
      printf("%d> %.4s", cycle,
        &"LLA ,IMM ,JMP ,JSR ,BZ  ,BNZ ,ENT ,ADJ ,LEV ,LI  ,LC  ,SI  ,SC  ,PSH ,"
         "OR  ,XOR ,AND ,EQ  ,NE  ,LT  ,GT  ,LE  ,GE  ,SHL ,SHR ,ADD ,SUB ,MUL ,DIV ,MOD ,"
         "OPEN,READ,CLOS,PRTF,MALC,FREE,MSET,MCMP,EXIT,"[i * 5]);
      if (i <= ADJ) printf(" %d\n", *pc); else printf("\n");
    }
    if      (i == LLA) a = (int)(bp + *pc++);                             // load local address 載入區域變數
    else if (i == IMM) a = *pc++;                                         // load global address or immediate 載入全域變數或立即值
    else if (i == JMP) pc = (int *)*pc;                                   // jump               躍躍指令
    else if (i == JSR) { *--sp = (int)(pc + 1); pc = (int *)*pc; }        // jump to subroutine 跳到副程式
    else if (i == BZ)  pc = a ? pc + 1 : (int *)*pc;                      // branch if zero     if (a==0) goto m[pc]
    else if (i == BNZ) pc = a ? (int *)*pc : pc + 1;                      // branch if not zero if (a!=0) goto m[pc]
    else if (i == ENT) { *--sp = (int)bp; bp = sp; sp = sp - *pc++; }     // enter subroutine   進入副程式
    else if (i == ADJ) sp = sp + *pc++;                                   // stack adjust       調整堆疊
    else if (i == LEV) { sp = bp; bp = (int *)*sp++; pc = (int *)*sp++; } // leave subroutine   離開副程式
    else if (i == LI)  a = *(int *)a;                                     // load int           載入整數
    else if (i == LC)  a = *(char *)a;                                    // load char          載入字元
    else if (i == SI)  *(int *)*sp++ = a;                                 // store int          儲存整數
    else if (i == SC)  a = *(char *)*sp++ = a;                            // store char         儲存字元
    else if (i == PSH) *--sp = a;                                         // push               推入堆疊

    else if (i == OR)  a = *sp++ |  a; // a = a OR *sp
    else if (i == XOR) a = *sp++ ^  a; // a = a XOR *sp
    else if (i == AND) a = *sp++ &  a; // ...
    else if (i == EQ)  a = *sp++ == a;
    else if (i == NE)  a = *sp++ != a;
    else if (i == LT)  a = *sp++ <  a;
    else if (i == GT)  a = *sp++ >  a;
    else if (i == LE)  a = *sp++ <= a;
    else if (i == GE)  a = *sp++ >= a;
    else if (i == SHL) a = *sp++ << a;
    else if (i == SHR) a = *sp++ >> a;
    else if (i == ADD) a = *sp++ +  a;
    else if (i == SUB) a = *sp++ -  a;
    else if (i == MUL) a = *sp++ *  a;
    else if (i == DIV) a = *sp++ /  a;
    else if (i == MOD) a = *sp++ %  a;

    else if (i == OPEN) a = open((char *)sp[1], *sp); // 開檔
    else if (i == READ) a = read(sp[2], (char *)sp[1], *sp); // 讀檔
    else if (i == CLOS) a = close(*sp); // 關檔
    else if (i == PRTF) { t = sp + pc[1]; a = printf((char *)t[-1], t[-2], t[-3], t[-4], t[-5], t[-6]); } // printf("....", a, b, c, d, e)
    else if (i == MALC) a = (int)malloc(*sp); // 分配記憶體
    else if (i == FREE) free((void *)*sp); // 釋放記憶體
    else if (i == MSET) a = (int)memset((char *)sp[2], sp[1], *sp); // 設定記憶體
    else if (i == MCMP) a = memcmp((char *)sp[2], (char *)sp[1], *sp); // 比較記憶體
    else if (i == EXIT) { printf("exit(%d) cycle = %d\n", *sp, cycle); return *sp; } // EXIT 離開
    else { printf("unknown instruction = %d! cycle = %d\n", i, cycle); return -1; } // 錯誤處理
  }
}

// tokens and classes (operators last and in precedence order) (按優先權順序排列)
enum { // token : 0-127 直接用該字母表達， 128 以後用代號。
  Num = 128, Fun, Sys, Glo, Loc, Id,
  Char, Else, Enum, If, Int, Return, Sizeof, While,
  Assign, Cond, Lor, Lan, Or, Xor, And, Eq, Ne, Lt, Gt, Le, Ge, Shl, Shr, Add, Sub, Mul, Div, Mod, Inc, Dec, Brak
};

// types (支援型態，只有 int, char, pointer)
enum { CHAR, INT, PTR };

#define POOL_SIZE 256*1024

char source[POOL_SIZE] = {0};
int code[POOL_SIZE] = {0}, stack[POOL_SIZE] = {0};
char data[POOL_SIZE] = {0};
char *datap = data;

char *p, *lp; // current position in source code (p: 目前原始碼指標, lp: 上一行原始碼指標)

int *e, *le,  // current position in emitted code (e: 目前機器碼指標, le: 上一行機器碼指標)
    loc,      // local variable offset (區域變數的位移)
    line;     // current line number (目前行號)

typedef struct token_t {
   int tk, class, type, val, len;
   char *name;
} token_t;

token_t sym[POOL_SIZE] = {
  {.tk=Id, .class=Sys, .name="printf", .type=INT, .val=PRTF, .len=6},
  {.tk=Int, .class=0, .name="int", .type=INT, .val=0, .len=3},
  {.tk=Char, .class=0, .name="char", .type=CHAR, .val=0, .len=4},
  {.tk=While, .class=0, .name="while", .type=0, .val=0, .len=5},
};

void sym_dump(token_t *sym) {
  token_t *id = sym;
  while (id->name) {
    printf("sym[%d]: %-8.*s len=%d tk=%d class=%d type=%d val=%d\n", id-sym, id->len, id->name, id->len, id->tk, id->class, id->type, id->val);
    id++;
  }
}

token_t t; // 初始化 token

void next() // 詞彙解析 lexer
{
  memset(&t, 0, sizeof(token_t));
  t.class = Glo;
  t.len = 1;
  while (t.tk = *p) {
    t.name = p;
    ++p;
    if (t.tk == '\n') { // 換行
      if (src) {
        printf("%d: %.*s", line, p - lp, lp); // 印出該行
        lp = p; // lp = p = 新一行的原始碼開頭
        while (le < e) { // 印出上一行的所有目的碼
          printf("%8.4s", &"LLA ,IMM ,JMP ,JSR ,BZ  ,BNZ ,ENT ,ADJ ,LEV ,LI  ,LC  ,SI  ,SC  ,PSH ,"
                           "OR  ,XOR ,AND ,EQ  ,NE  ,LT  ,GT  ,LE  ,GE  ,SHL ,SHR ,ADD ,SUB ,MUL ,DIV ,MOD ,"
                           "OPEN,READ,CLOS,PRTF,MALC,FREE,MSET,MCMP,EXIT,"[*++le * 5]);
          if (*le <= ADJ) printf(" %d\n", *++le); else printf("\n"); // LLA ,IMM ,JMP ,JSR ,BZ  ,BNZ ,ENT ,ADJ 有一個參數。
        }
      }
      ++line;
    }
    else if (t.tk == '#') { // 取得 #include <stdio.h> 這類的一整行
      while (*p != 0 && *p != '\n') ++p;
    }
    else if ((t.tk >= 'a' && t.tk <= 'z') || (t.tk >= 'A' && t.tk <= 'Z') || t.tk == '_') { // 取得變數名稱
      t.name = p-1; // 變數名稱
      while ((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z') || (*p >= '0' && *p <= '9') || *p == '_')
        p++;
      t.len = p - t.name;
      token_t *id = sym;
      while (id->name) { // 檢查符號是否存在
        if (t.len == id->len && !memcmp(id->name, t.name, p - t.name)) { t = *id; return; } // 有找到的話就傳回
        id++; // 找下一個符號
      }
      t.tk = Id;
      *id = t;
      return;
    }
    else if (t.tk >= '0' && t.tk <= '9') { // 取得數字串
      int ival;
      if (ival = t.tk - '0') { while (*p >= '0' && *p <= '9') ival = ival * 10 + *p++ - '0'; } // 十進位
      else if (*p == 'x' || *p == 'X') { // 十六進位
        while ((t.tk = *++p) && ((t.tk >= '0' && t.tk <= '9') || (t.tk >= 'a' && t.tk <= 'f') || (t.tk >= 'A' && t.tk <= 'F'))) // 16 進位
          ival = ival * 16 + (t.tk & 15) + (t.tk >= 'A' ? 9 : 0);
      }
      else { while (*p >= '0' && *p <= '7') ival = ival * 8 + *p++ - '0'; } // 八進位
      t.tk = Num; // token = Number
      t.val = ival;
      return;
    }
    else if (t.tk == '/') {
      if (*p == '/') { // 註解
        ++p;
        while (*p != 0 && *p != '\n') ++p; // 略過註解
      }
      else { // 除法
        t.tk = Div;
        return;
      }
    }
    else if (t.tk == '\'' || t.tk == '"') { // 字元或字串
      char *pp = datap;
      while (*p != 0 && *p != t.tk) {
        int ival;
        if ((ival = *p++) == '\\') {
          if ((ival = *p++) == 'n') ival = '\n'; // 處理 \n 的特殊情況
        }
        if (t.tk == '"') *datap++ = ival; // 把字串塞到資料段裏
        t.val = ival;
      }
      if (t.tk == '"') *datap++ = '\0';
      ++p;
      if (t.tk == '"') t.val = (int)pp; else t.tk = Num; // (若是字串) ? (ival = 字串 (在資料段中的) 指標) : (字元值)
      return;
    } // 以下為運算元 =+-!<>|&^%*[?~, ++, --, !=, <=, >=, ||, &&, ~  ;{}()],:
    else if (t.tk == '=') { if (*p == '=') { ++p; t.tk = Eq; } else t.tk = Assign; return; }
    else if (t.tk == '+') { if (*p == '+') { ++p; t.tk = Inc; } else t.tk = Add; return; }
    else if (t.tk == '-') { if (*p == '-') { ++p; t.tk = Dec; } else t.tk = Sub; return; }
    else if (t.tk == '!') { if (*p == '=') { ++p; t.tk = Ne; } return; }
    else if (t.tk == '<') { if (*p == '=') { ++p; t.tk = Le; } else if (*p == '<') { ++p; t.tk = Shl; } else t.tk = Lt; return; }
    else if (t.tk == '>') { if (*p == '=') { ++p; t.tk = Ge; } else if (*p == '>') { ++p; t.tk = Shr; } else t.tk = Gt; return; }
    else if (t.tk == '|') { if (*p == '|') { ++p; t.tk = Lor; } else t.tk = Or; return; }
    else if (t.tk == '&') { if (*p == '&') { ++p; t.tk = Lan; } else t.tk = And; return; }
    else if (t.tk == '^') { t.tk = Xor; return; }
    else if (t.tk == '%') { t.tk = Mod; return; }
    else if (t.tk == '*') { t.tk = Mul; return; }
    else if (t.tk == '[') { t.tk = Brak; return; }
    else if (t.tk == '?') { t.tk = Cond; return; }
    else if (t.tk == '~' || t.tk == ';' || t.tk == '{' || t.tk == '}' || t.tk == '(' || t.tk == ')' || t.tk == ']' || t.tk == ',' || t.tk == ':') return;
  }
}

int lex() {
  p = lp = source;
  datap = data;
  line = 1;
  while (1) {
    next();
    if (t.tk == 0) break;
    printf("p=%p tk=%d sym=%8.4s\n", p, t.tk, t.name);
  }
}

int ty;

// CALL(id) = (E*)
// F = (E) | Number | Id | CALL
// E = F (op E)*
// ASSIGN = id '=' E
// WHILE = while (E) STMT
// STMT = WHILE | BLOCK | CALL | ASSIGN
// STMTS = STMT*
// BLOCK = { STMT* }

int expr() {
  int *d;

  if (!t.tk) { printf("%d: unexpected eof in expression\n", line); exit(-1); } // EOF
  else if (t.tk == Num) { *++e = IMM; *++e = t.val; next(); t.type = INT; } // 數值
  else if (t.tk == '"') { // 字串
    *++e = IMM; *++e = t.val; next();
    while (t.tk == '"') next();
    datap = (char *)((int)datap + sizeof(int) & -sizeof(int)); t.type = PTR; // 用 int 為大小對齊 ??
  }
  else if (t.tk == Id) { // 處理 id ...
    token_t id = t; next();
    if (t.tk == Assign) {
      next();
      if (*e == LC || *e == LI) *e = PSH; else { printf("%d: bad lvalue in assignment\n", line); exit(-1); }
      expr(); *++e = (ty == CHAR) ? SC : SI;
    } else if (t.tk == '(') { // id (args) ，這是 call
      int arg_count = 0;
      next();
      while (t.tk != ')') { expr(); *++e = PSH; ++arg_count; if (t.tk == ',') next(); } // 推入 arg
      next();
      if (id.class == Sys) *++e = id.val; // token 是系統呼叫
      else if (id.class == Fun) { *++e = JSR; *++e = id.val; } // token 是自訂函數，用 JSR : jump to subroutine 指令呼叫
      else { printf("%d: bad function call\n", line); exit(-1); }
      if (arg_count) { *++e = ADJ; *++e = arg_count; } // 有參數，要調整堆疊  (ADJ : stack adjust)
      ty = id.type;
    }
    else if (id.class == Num) { *++e = IMM; *++e = id.val; ty = INT; } // 該 id 是數值
    else {
      if (id.class == Loc) { *++e = LLA; *++e = loc - id.val; } // 該 id 是區域變數，載入區域變數 (LLA : load local address)
      else if (id.class == Glo) { *++e = IMM; *++e = id.val; }  // 該 id 是全域變數，載入該全域變數 (IMM : load global address or immediate 載入全域變數或立即值)
      else { printf("%d: undefined variable\n", line); exit(-1); }
      *++e = ((ty = id.type) == CHAR) ? LC : LI; // LI  : load int, LC  : load char
    }
  }
  else if (t.tk == '(') { // (E) : 有括號的運算式 ...
    next();
    expr(); // 處理 (E) 中的 E      (E 運算式必須能處理 (t=x) op y 的情況，所以用 expr(Assign))
    if (t.tk == ')') next(); else { printf("%d: close paren expected\n", line); exit(-1); }
  }
  else {
    printf("%d: bad expression\n", line); exit(-1);
  }

  if (t.tk == Lor) { next(); *++e = BNZ; d = ++e; expr(Lan); *d = (int)(e + 1); ty = INT; }
  else if (t.tk == Lan) { next(); *++e = BZ;  d = ++e; expr(Or);  *d = (int)(e + 1); ty = INT; }
  else if (t.tk == Or)  { next(); *++e = PSH; expr(Xor); *++e = OR;  ty = INT; }
  else if (t.tk == Xor) { next(); *++e = PSH; expr(And); *++e = XOR; ty = INT; }
  else if (t.tk == And) { next(); *++e = PSH; expr(Eq);  *++e = AND; ty = INT; }
  else if (t.tk == Eq)  { next(); *++e = PSH; expr(Lt);  *++e = EQ;  ty = INT; }
  else if (t.tk == Ne)  { next(); *++e = PSH; expr(Lt);  *++e = NE;  ty = INT; }
  else if (t.tk == Lt)  { next(); *++e = PSH; expr(Shl); *++e = LT;  ty = INT; }
  else if (t.tk == Gt)  { next(); *++e = PSH; expr(Shl); *++e = GT;  ty = INT; }
  else if (t.tk == Le)  { next(); *++e = PSH; expr(Shl); *++e = LE;  ty = INT; }
  else if (t.tk == Ge)  { next(); *++e = PSH; expr(Shl); *++e = GE;  ty = INT; }
  else if (t.tk == Shl) { next(); *++e = PSH; expr(Add); *++e = SHL; ty = INT; }
  else if (t.tk == Shr) { next(); *++e = PSH; expr(Add); *++e = SHR; ty = INT; }
  else if (t.tk == Lor) { next(); *++e = BNZ; d = ++e; expr(Lan); *d = (int)(e + 1); ty = INT; }
  else if (t.tk == Lan) { next(); *++e = BZ;  d = ++e; expr(Or);  *d = (int)(e + 1); ty = INT; }
  else if (t.tk == Or)  { next(); *++e = PSH; expr(Xor); *++e = OR;  ty = INT; }
  else if (t.tk == Xor) { next(); *++e = PSH; expr(And); *++e = XOR; ty = INT; }
  else if (t.tk == And) { next(); *++e = PSH; expr(Eq);  *++e = AND; ty = INT; }
  else if (t.tk == Eq)  { next(); *++e = PSH; expr(Lt);  *++e = EQ;  ty = INT; }
  else if (t.tk == Ne)  { next(); *++e = PSH; expr(Lt);  *++e = NE;  ty = INT; }
  else if (t.tk == Lt)  { next(); *++e = PSH; expr(Shl); *++e = LT;  ty = INT; }
  else if (t.tk == Gt)  { next(); *++e = PSH; expr(Shl); *++e = GT;  ty = INT; }
  else if (t.tk == Le)  { next(); *++e = PSH; expr(Shl); *++e = LE;  ty = INT; }
  else if (t.tk == Ge)  { next(); *++e = PSH; expr(Shl); *++e = GE;  ty = INT; }
  else if (t.tk == Shl) { next(); *++e = PSH; expr(Add); *++e = SHL; ty = INT; }
  else if (t.tk == Shr) { next(); *++e = PSH; expr(Add); *++e = SHR; ty = INT; }
  else if (t.tk == Add) { next(); *++e = PSH; expr(Mul); *++e = ADD; ty = INT; }
  else if (t.tk == Sub) { next(); *++e = PSH; expr(Sub); *++e = SUB; ty = INT; }
  else if (t.tk == Mul) { next(); *++e = PSH; expr(Inc); *++e = MUL; ty = INT; }
  else if (t.tk == Div) { next(); *++e = PSH; expr(Inc); *++e = DIV; ty = INT; }
  else if (t.tk == Mod) { next(); *++e = PSH; expr(Inc); *++e = MOD; ty = INT; }
  else if (t.tk == Inc || t.tk == Dec) {
    if (*e == LC) { *e = PSH; *++e = LC; }
    else if (*e == LI) { *e = PSH; *++e = LI; }
    else { printf("%d: bad lvalue in post-increment\n", line); exit(-1); }
    *++e = PSH; *++e = IMM; *++e = (ty > PTR) ? sizeof(int) : sizeof(char);
    *++e = (t.tk == Inc) ? ADD : SUB;
    *++e = (ty == CHAR) ? SC : SI;
    *++e = PSH; *++e = IMM; *++e = (ty > PTR) ? sizeof(int) : sizeof(char);
    *++e = (t.tk == Inc) ? SUB : ADD;
    next();
  }
  else {

  }
}

int stmt() {
  int *a, *b;
  if (t.tk == Int || t.tk == Char) { // 宣告 ex: int a, b;
    next();
    // while (t.tk != ';') {
      ty = (t.tk == Int) ? INT : CHAR;
      next();
      while (t.tk == ',') {
        next();
        if (t.tk != Id) { printf("%d: bad declaration\n", line); exit(-1); }
        t.type = ty;
        next();
      }
      if (t.tk == ';') next(); else { printf("%d: semicolon expected\n", line); exit(-1); }
    // }
  } else if (t.tk == While) { // while 語句
    next();
    a = e + 1;
    // if (t.tk == '(') next(); else { printf("%d: open paren expected\n", line); exit(-1); }
    expr();
    // if (t.tk == ')') next(); else { printf("%d: close paren expected\n", line); exit(-1); }
    *++e = BZ; b = ++e;
    stmt();
    *++e = JMP; *++e = (int)a;
    *b = (int)(e + 1);
  } else if (t.tk == '{') { // 區塊 {...}
    next();
    while (t.tk != '}') stmt();
    next();
  } else if (t.tk == ';') { // ; 空陳述
    next();
  } else { // ASSIGN or EXP
    if (t.tk == Id) {
      next();
      if (t.tk == '(') {

      } else if (t.tk == Assign) {

      } else {
        expr();
      }
    }
    if (t.tk == ';') next(); else { printf("stmt: %d: semicolon expected\n", line); exit(-1); }
  }
}

int prog() {
  while (t.tk) {
    stmt();
  }
}

int compile() {
  e = le = code-1;
  p = lp = source;
  datap = data;
  line = 1;
  next();
  prog();
}

int main(int32_t argc, char *argv[]) // 主程式
{
  --argc; ++argv;
  if (argc > 0 && **argv == '-' && (*argv)[1] == 's') { src = 1; --argc; ++argv; }
  if (argc > 0 && **argv == '-' && (*argv)[1] == 'd') { debug = 1; --argc; ++argv; }
  if (argc < 1) { printf("usage: c4 [-s] [-d] file ...\n"); return -1; }
  printf("src=%d\n", src);
  int *pc, *bp, *sp;
  int fd = open(*argv, 0);
  if (fd < 0) { printf("could not open(%s)\n", *argv); return -1; }
  int len = read(fd, source, POOL_SIZE-1);
  if (len <= 0) { printf("read() returned %d\n", len); return -1; }

  // printf("============ symbols ==========\n");
  // sym_dump(sym);
  // printf("============ lex ==============\n");
  // lex();
  // printf("============ symbols ==========\n");
  // sym_dump(sym);
  printf("============ compile ==========\n");
  compile();
  *++e = EXIT;
  pc = code;
  bp = sp = (int *)((int)stack + POOL_SIZE); // setup stack
  return run(pc, bp, sp);
}
