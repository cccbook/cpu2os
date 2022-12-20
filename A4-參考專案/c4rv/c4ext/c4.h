#ifndef __C4_H__
#define __C4_H__

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <unistd.h>
#include <fcntl.h>
#include "bc.h"

// tokens and classes (operators last and in precedence order) (按優先權順序排列)
enum { // token : 0-127 直接用該字母表達， 128 以後用代號。
  Num = 128, Fun, Sys, Glo, Loc, Id, Char, Else, Enum, If, Int, Return, Sizeof, While,
  Assign, Cond, Lor, Lan, Or, Xor, And, Eq, Ne, Lt, Gt, Le, Ge, Shl, Shr, Add, Sub, 
  Mul, Div, Mod, Inc, Dec, Brak
};

// types (支援型態，只有 int, char, pointer)
enum { CHAR, INT, PTR };

// 因為沒有 struct，所以使用 offset 代替，例如 id[Tk] 代表 id.Tk (token), id[Hash] 代表 id.Hash, id[Name] 代表 id.Name, .....
// identifier offsets (since we can't create an ident struct)
enum { Tk, Hash, Name, Class, Type, Val, HClass, HType, HVal, Idsz }; // HClass, HType, HVal 是暫存的備份 ???

// opcodes (機器碼的 op)
enum { IMM ,LVAR ,GVAR, STR, JMP ,JSR ,BZ  ,BNZ ,ENT ,ADJ ,LEV ,LI  ,LC  ,SI  ,SC  ,PSH ,
       OR  ,XOR ,AND ,EQ  ,NE  ,LT  ,GT  ,LE  ,GE  ,SHL ,SHR ,ADD ,SUB ,MUL ,DIV ,MOD ,
       OPEN,READ,WRIT,CLOS,PRTF,SPRT,SSCN,GETS,MALC,FREE,MSET,MCMP,MCPY,EXIT };

enum { O_READ=0, O_WRITE = 769 }; // O_READ: 讀取模式, O_WRITE: 寫入模式, 769 = O_WRONLY|O_CREAT|O_TRUNC
enum { RCode, RData }; // RCode: 程式重定位, RData 資料重定位。

#define open(fname, flag, mode) (int)fopen(fname, (flag==0)?"rb":"wb")
#define read(fd, buf, size)  (int)fread(buf, 1, size, (FILE*)fd)
#define write(fd, buf, size) (int)fwrite(buf, 1, size, (FILE*)fd)
#define close(fd) fclose((FILE*)fd)

extern char *symClassNames; // "Fun , Sys , Glo ,"

extern char *ops; // 代表以上運算列表的字串 "LVAR  IMM  ...."。 (最多四個字)

extern char *p, *lp; // current position in source code (p: 目前原始碼指標, lp: 上一行原始碼指標)
extern int *pc, // code: 程式段, pc: 程式計數器
    *e, *le,  // current position in emitted code (e: 目前機器碼指標, le: 上一行機器碼指標)
    *id,      // currently parsed identifier (id: 目前的 id)
    *sym,     // symbol table (simple list of identifiers) (符號表)
    tk,       // current token (目前 token)
    ival,     // current token value (目前的 token 值)
    ty,       // current expression type (目前的運算式型態)
    loc,      // local variable offset (區域變數的位移)
    line,     // current line number (目前行號)
    src,      // print source and assembly flag (印出原始碼)
    debug,    // print executed instructions (印出執行指令 -- 除錯模式)
    poolsz;   // 分配空間大小

extern char *cFile, *oFile;
extern int *code, *stack,  // code:程式段, stack:堆疊段
    *rel, *relp,    // rel: 重定位紀錄
    output;         // output: 輸出目的檔 
extern char *symstab, *symstabp, // 符號字串表與尾端指標
     *stab, *stabp, // 符號字串表與尾端指標
     *source, // source:c 語言原始碼,
     *data, *datap; // datap/bss pointer (資料段機器碼指標)
extern char *data0; // data0:資料段的起始點 (重定位會用)
extern int *code0; // code0:程式段的起始點 (重定位會用)
extern char *obj;          // 目的碼
extern int output;         // output: 輸出目的檔

extern void *malloc0(int size);
extern void init();
extern int compile(char *file);
extern int vm_run(int *pc, int *bp, int *sp, char *data, char *stab);
extern int vm_main(int *pc, int argc, char **argv);

#endif
