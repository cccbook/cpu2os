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
#define int long long // 64 bit 電腦

char *p, *lp, // current position in source code (p: 目前原始碼指標, lp: 上一行原始碼指標)
     *data, *datap; // datap/bss pointer (資料段機器碼指標，包含《字串常數+全域變數空間》)。

int *e, *le,  // current position in emitted code (e: 目前機器碼指標, le: 上一行機器碼指標)
    *id,      // currently parsed identifier (id: 目前的 id)
    *sym,     // symbol table (simple list of identifiers) (符號表)
    tk,       // current token (目前 token)
    ival,     // current token value (目前的 token 值)
    ty,       // current expression type (目前的運算式型態)
    loc,      // local variable offset (區域變數的位移)
    line,     // current line number (目前行號)
    table,    // print table (印出符號表，字串表)
    src,      // print source and assembly flag (印出原始碼)
    debug;    // print executed instructions (印出執行指令 -- 除錯模式)

// tokens and classes (operators last and in precedence order) (按優先權順序排列)
enum { // token : 0-127 直接用該字母表達， 128 以後用代號。
  Num = 128, Fun, Sys, Glo, Loc, Id,
  Char, Else, Enum, If, Int, Return, Sizeof, While,
  Assign, Cond, Lor, Lan, Or, Xor, And, Eq, Ne, Lt, Gt, Le, Ge, Shl, Shr, Add, Sub, Mul, Div, Mod, Inc, Dec, Brak
};

// opcodes (機器碼的 op)
enum { LLA, IMM ,STR ,LGA ,JMP ,JSR ,BZ  ,BNZ ,ENT ,ADJ ,LEV ,LI  ,LC  ,SI  ,SC  ,PSH ,
       OR  ,XOR ,AND ,EQ  ,NE  ,LT  ,GT  ,LE  ,GE  ,SHL ,SHR ,ADD ,SUB ,MUL ,DIV ,MOD ,
       OPEN,READ,CLOS,PRTF,MALC,FREE,MSET,MCMP,EXIT };

// types (支援型態，只有 int, char, pointer)
enum { CHAR, INT, PTR };

// 因為沒有 struct，所以使用 offset 代替，例如 id[Tk] 代表 id.Tk (token), id[Hash] 代表 id.Hash, id[Name] 代表 id.Name, .....
// identifier offsets (since we can't create an ident struct)
enum { Tk, Hash, Name, Class, Type, Val, HClass, HType, HVal, Idsz }; // HClass, HType, HVal 是暫存的備份 ???

void printId(char *p) {
  while ((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z') || (*p >= '0' && *p <= '9') || *p == '_')
    printf("%c", *p++);
}

void printOp(int op) {
    printf("%.4s", &"LLA ,IMM ,STR ,LGA ,JMP ,JSR ,BZ  ,BNZ ,ENT ,ADJ ,LEV ,LI  ,LC  ,SI  ,SC  ,PSH ,"
                    "OR  ,XOR ,AND ,EQ  ,NE  ,LT  ,GT  ,LE  ,GE  ,SHL ,SHR ,ADD ,SUB ,MUL ,DIV ,MOD ,"
                    "OPEN,READ,CLOS,PRTF,MALC,FREE,MSET,MCMP,EXIT,"[op * 5]);
}

void printTk(int tk) {
  printf("tk=%3d ", tk);
  if (tk < 128) { printf("%c", (char) tk); return; } // 單一字元 token
  if (tk == Id) { printId(id[Name]); return; }
  if (tk > Char && tk < While) { printId(id[Name]); return; }
}

int poolsz;

void symDump() {
  int *sid, *symEnd, i;
  printf("============ symbol table ===================\n");
  sid = sym;
  i = 0;
  while (sid[Tk]) {
    if (sid[Class]==Loc)      { printf("%2d:loc:      ", i); printId(sid[Name]); printf("\n"); }
    else if (sid[Class]==Num) { printf("%2d:num:      ", i); printId(sid[Name]); printf("\n"); }
    else if (sid[Class]==Sys) { printf("%2d:system:   tk=%2d val=%2d ", i, sid[Tk], sid[Val]); printId(sid[Name]); printf("\n"); }
    else if (sid[Class]==Glo) { printf("%2d:global:   ", i); printId(sid[Name]); printf(" at data[%d]\n", (char*)sid[Val]-data); }
    else if (sid[Class]==Fun) { printf("%2d:function: ", i); printId(sid[Name]); printf("\n"); }
    else if (sid[Class]==Id)  { printf("%2d:id:       tk=%2d ", i, sid[Tk]); printId(sid[Name]); printf("\n"); }
    else                      { printf("%2d:keyword:  tk=%2d ", i, sid[Tk]); printId(sid[Name]); printf("\n"); }
    sid = sid + Idsz; 
    i ++;
  }
}

void next() // 詞彙解析 lexer
{
  char *pp; int op, arg;

  while (tk = *p) {
    ++p;
    if (tk == '\n') { // 換行
      if (src) {
        printf("%d: %.*s", line, p - lp, lp); // 印出該行
        lp = p; // lp = p = 新一行的原始碼開頭
        while (le < e) { // 印出上一行的所有目的碼
          op = *++le;
          printf("\t"); printOp(op);
          if (op <= ADJ) {
            arg = *++le;
            printf(" %d", arg); 
            if (op == STR) printf(" // string:%d\n", (char*)arg-data);
            if (op == LGA) printf(" // gvar:%d\n",  (char*)arg-data);
            printf("\n");
          } else
            printf("\n"); // LLA ,IMM ,STR ,LGA,JMP ,JSR ,BZ  ,BNZ ,ENT ,ADJ 有一個參數。
        }
      }
      ++line;
    }
    else if (tk == '#') { // 取得 #include <stdio.h> 這類的一整行
      while (*p != 0 && *p != '\n') ++p;
    }
    else if ((tk >= 'a' && tk <= 'z') || (tk >= 'A' && tk <= 'Z') || tk == '_') { // 取得變數名稱
      pp = p - 1;
      while ((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z') || (*p >= '0' && *p <= '9') || *p == '_')
        tk = tk * 147 + *p++;  
      tk = (tk << 7) + (p - pp); // 最後的雜湊值 tk; 原本是 << 6 ; 2^6=64 沒超過 128，怕會和 ASCII 衝碼
      id = sym; // 從 sym 表頭開始
      while (id[Tk]) { // 找到 hash 為 tk 的那個
        if (tk == id[Hash] && !memcmp((char *)id[Name], pp, p - pp)) { tk = id[Tk]; return; } // 有找到，變數出現過，找到該 id 了
        id = id + Idsz; // 前進到下一格
      }
      // 否則為新 id，設定該 id 的表格內容。
      id[Name] = (int)pp; // id.Name = ptr(變數名稱)
      id[Hash] = tk;      // id.Hash = 雜湊值
      tk = id[Tk] = Id;   // token = id.Tk = Id
      return; // 將 id 交給 parser 處理。
    }
    else if (tk >= '0' && tk <= '9') { // 取得數字串
      if (ival = tk - '0') { while (*p >= '0' && *p <= '9') ival = ival * 10 + *p++ - '0'; } // 十進位
      else if (*p == 'x' || *p == 'X') { // 十六進位
        while ((tk = *++p) && ((tk >= '0' && tk <= '9') || (tk >= 'a' && tk <= 'f') || (tk >= 'A' && tk <= 'F'))) // 16 進位
          ival = ival * 16 + (tk & 15) + (tk >= 'A' ? 9 : 0);
      }
      else { while (*p >= '0' && *p <= '7') ival = ival * 8 + *p++ - '0'; } // 八進位
      tk = Num; // token = Number
      return;
    }
    else if (tk == '/') {
      if (*p == '/') { // 註解
        ++p;
        while (*p != 0 && *p != '\n') ++p; // 略過註解
      }
      else { // 除法
        tk = Div;
        return;
      }
    }
    else if (tk == '\'' || tk == '"') { // 字元或字串
      pp = datap;
      while (*p != 0 && *p != tk) {
        ival = *p++;
        if (ival == '\\') {
          if ((ival = *p++) == 'n') ival = '\n'; // 處理 \n 的特殊情況
        }
        if (tk == '"') // 是字串 "..." ，非 '..'
          *datap++ = ival; // 把目前掃到的字塞到資料段裏
      }
      ++p;
      if (tk == '"') ival = (int)pp; else tk = Num; // (若是字串) ? (ival = 字串 (在資料段中的) 指標) : (字元值)
      // 注意，字串不會保留 " 符號在 datap 段中
      // 問題：字串會塞結尾的 \0 進資料段嗎？在上述程式中沒有看到? (原本初始化就有設 0，而且 expr 函數中會對齊，所以不用塞)
      return;
    } // 以下為運算元 =+-!<>|&^%*[?~, ++, --, !=, <=, >=, ||, &&, ~  ;{}()],:
    else if (tk == '=') { if (*p == '=') { ++p; tk = Eq; } else tk = Assign; return; }
    else if (tk == '+') { if (*p == '+') { ++p; tk = Inc; } else tk = Add; return; }
    else if (tk == '-') { if (*p == '-') { ++p; tk = Dec; } else tk = Sub; return; }
    else if (tk == '!') { if (*p == '=') { ++p; tk = Ne; } return; }
    else if (tk == '<') { if (*p == '=') { ++p; tk = Le; } else if (*p == '<') { ++p; tk = Shl; } else tk = Lt; return; }
    else if (tk == '>') { if (*p == '=') { ++p; tk = Ge; } else if (*p == '>') { ++p; tk = Shr; } else tk = Gt; return; }
    else if (tk == '|') { if (*p == '|') { ++p; tk = Lor; } else tk = Or; return; }
    else if (tk == '&') { if (*p == '&') { ++p; tk = Lan; } else tk = And; return; }
    else if (tk == '^') { tk = Xor; return; }
    else if (tk == '%') { tk = Mod; return; }
    else if (tk == '*') { tk = Mul; return; }
    else if (tk == '[') { tk = Brak; return; }
    else if (tk == '?') { tk = Cond; return; }
    else if (tk == '~' || tk == ';' || tk == '{' || tk == '}' || tk == '(' || tk == ')' || tk == ']' || tk == ',' || tk == ':') return;
  }
}

int lex() {
  while (1) {
    next();
    if (tk==0) break;
    printTk(tk);
    printf("\n");
  }
}

int main(int argc, char **argv) // 主程式
{
  int fd, ty, *idmain;
  int *pc, *bp, *sp;
  int i, *t;

  --argc; ++argv;
  if (argc > 0 && **argv == '-' && (*argv)[1] == 's') { src = 1; --argc; ++argv; }
  if (argc > 0 && **argv == '-' && (*argv)[1] == 'd') { debug = 1; --argc; ++argv; }
  if (argc > 0 && **argv == '-' && (*argv)[1] == 't') { table = 1; --argc; ++argv; }
  if (argc < 1) { printf("usage: c4 [-s] [-d] [-t] file ...\n"); return -1; }

  if ((fd = open(*argv, 0)) < 0) { printf("could not open(%s)\n", *argv); return -1; }

  poolsz = 256*1024; // arbitrary size
  if (!(sym = malloc(poolsz))) { printf("could not malloc(%d) symbol area\n", poolsz); return -1; } // 符號段
  if (!(le = e = malloc(poolsz))) { printf("could not malloc(%d) text area\n", poolsz); return -1; } // 程式段
  if (!(data = datap = malloc(poolsz))) { printf("could not malloc(%d) datap area\n", poolsz); return -1; } // 資料段
  if (!(sp = malloc(poolsz))) { printf("could not malloc(%d) stack area\n", poolsz); return -1; }  // 堆疊段

  if (!(lp = p = malloc(poolsz))) { printf("could not malloc(%d) source area\n", poolsz); return -1; }
  if ((i = read(fd, p, poolsz-1)) <= 0) { printf("read() returned %d\n", i); return -1; }
  p[i] = 0; // 設定程式 p 字串結束符號 \0
  close(fd);

  memset(sym,  0, poolsz);
  memset(e,    0, poolsz);
  memset(datap, 0, poolsz);
  p = "char else enum if int return sizeof while "
      "open read close printf malloc free memset memcmp exit void main";
  i = Char; while (i <= While) { next(); id[Tk] = i++; } // add keywords to symbol table
  i = OPEN; while (i <= EXIT) { next(); id[Class] = Sys; id[Type] = INT; id[Val] = i++; } // add library to symbol table
  next(); id[Tk] = Char; // handle void type

  p=lp;
  lex();
  symDump();
  return;
}