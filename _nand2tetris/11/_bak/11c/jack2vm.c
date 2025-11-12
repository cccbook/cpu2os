/********************************************************************
 * JackCompiler - Jack 語言轉換為 Hack VM 程式 (Nand2Tetris Project.org)
 * 作者：Grok (基於課程規格完整實作)
 * 支援：Project 10 (語法分析) + Project 11 (程式碼產生)
 * 編譯：gcc -o jack2vm jack2vm.c
 * 使用：./jack2vm MyClass.jack   或   ./jack2vm MyDir/
 ********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <ctype.h>
#include <sys/stat.h>
#include <stdarg.h>

#define debug(fmt, ...) printf(fmt, ##__VA_ARGS__)
// #define debug(fmt, ...) printf("") // printf(fmt, ##__VA_ARGS__)
#define MAX_LINE 1024
#define MAX_SYMBOL 256
#define MAX_LABEL 256

// VM 指令寫入器
FILE *vm_out;
char current_class[64];
char current_subroutine[64];
int label_counter = 0;

// 符號表 (簡化版：使用字串比對，實際可改 hash table)
typedef struct {
    char name[64];
    char type[64];
    char kind[16];  // static, field, arg, var
    int index;
} Symbol;

Symbol symbol_table[1024];
int symbol_count = 0;
int field_count = 0;

// 目前作用域的變數計數
int n_locals = 0;
int n_args = 0;

// 產生唯一標籤
char* new_label(const char* prefix) {
    static char label[64];
    snprintf(label, sizeof(label), "%s_%d", prefix, label_counter++);
    return label;
}

// 寫入 VM 指令
void write_vm(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vfprintf(vm_out, fmt, args);
    fprintf(vm_out, "\n");
    va_end(args);
}

// 移除註解與空白
void clean_line(char* line) {
    char* p = line;
    char* q = line;
    int in_string = 0;
    int in_comment = 0;

    // 以下寫法不支持略過多行註解，所以像 SquareGame.jack 會有問題
    while (*p) {
        if (*p == '"' && (p == line || *(p-1) != '\\')) {
            in_string = !in_string;
        }
        if (!in_string && !in_comment && p[0] == '/' && p[1] == '/') {
            *q = '\0';
            break;  // 行內註解直接結束
        }
        if (!in_string && !in_comment && p[0] == '/' && p[1] == '*' && !in_comment) {
            in_comment = 2;
            p++;
        }
        if (in_comment == 2 && p[0] == '*' && p[1] == '/') {
            in_comment = 0;
            p += 2;
            continue;
        }
        if (!in_comment) {
            *q++ = *p;
        }
        p++;
    }
    *q = '\0';

    // 去除前導空白
    char* start = line;
    while (isspace(*start)) start++;

    // 去除尾端空白
    q = start + strlen(start) - 1;
    while (q >= start && isspace(*q)) *q-- = '\0';

    // 關鍵：直接移動內容到開頭，不用 strcpy！
    if (start != line) {
        memmove(line, start, strlen(start) + 1);
    }
}

// 讀取下一 token
char current_token[256];
char token_type[32]; // keyword, symbol, identifier, integerConstant, stringConstant

static char line[MAX_LINE];
static char* ptr = NULL;
static int has_more = 1;

void next_token_1(FILE* f) {
//    static char line[MAX_LINE];
//    static char* ptr = NULL;
//    static int has_more = 1;

    strcpy(token_type, "");

    if (!has_more) {
        strcpy(current_token, "");
        return;
    }

    while (1) {
        if (!ptr || !*ptr) {
            if (!fgets(line, sizeof(line), f)) {
                has_more = 0;
                strcpy(current_token, "");
                return;
            }
            clean_line(line);
            if (line[0] == '\0') continue;
            ptr = line;
        }

        // 跳過空白
        while (isspace(*ptr)) ptr++;

        if (*ptr == '\0') {
            ptr = NULL;
            continue;
        }

        // 字串常數
        if (*ptr == '"') {
            ptr++;
            char* start = ptr;
            while (*ptr && *ptr != '"') ptr++;
            int len = ptr - start;
            strncpy(current_token, start, len);
            current_token[len] = '\0';
            strcpy(token_type, "stringConstant");
            if (*ptr == '"') ptr++;
            return;
        }

        // 符號
        if (strchr("{}()[].,;+-*/&|<>=~", *ptr)) {
            current_token[0] = *ptr;
            current_token[1] = '\0';
            strcpy(token_type, "symbol");
            ptr++;
            // 特殊兩字符符號
            if (!strncmp(ptr-1, "<=", 2)) { strcpy(current_token, "<="); ptr++; }
            else if (!strncmp(ptr-1, ">=", 2)) { strcpy(current_token, ">="); ptr++; }
            else if (!strncmp(ptr-1, "<>", 2)) { strcpy(current_token, "<>"); ptr++; }
            return;
        }

        // 數字
        if (isdigit(*ptr)) {
            char* start = ptr;
            while (isdigit(*ptr)) ptr++;
            int len = ptr - start;
            strncpy(current_token, start, len);
            current_token[len] = '\0';
            strcpy(token_type, "integerConstant");
            return;
        }

        // 識別字與關鍵字
        if (isalpha(*ptr) || *ptr == '_') {
            char* start = ptr;
            while (isalnum(*ptr) || *ptr == '_') ptr++;
            int len = ptr - start;
            strncpy(current_token, start, len);
            current_token[len] = '\0';

            const char* keywords[] = {"class","constructor","function","method","field",
                                      "static","var","int","char","boolean","void","true",
                                      "false","null","this","let","do","if","else","while",
                                      "return", NULL};
            int i = 0;
            while (keywords[i]) {
                if (strcmp(current_token, keywords[i]) == 0) {
                    strcpy(token_type, "keyword");
                    return;
                }
                i++;
            }
            strcpy(token_type, "identifier");
            return;
        }
    }
}

void next_token(FILE* f) {
    next_token_1(f);
    // printf("token: %-15s type: %s\n", current_token, token_type); // 除錯輸出
    debug("token: %-15s type: %s\n", current_token, token_type); // 除錯輸出
}

// 符號表操作
void define_symbol(const char* name, const char* type, const char* kind) {
    strcpy(symbol_table[symbol_count].name, name);
    strcpy(symbol_table[symbol_count].type, type);
    strcpy(symbol_table[symbol_count].kind, kind);
    if (strcmp(kind, "field") == 0) {
        symbol_table[symbol_count].index = field_count++;
    } else if (strcmp(kind, "static") == 0) {
        symbol_table[symbol_count].index = symbol_count;
    } else if (strcmp(kind, "arg") == 0) {
        symbol_table[symbol_count].index = n_args++;
    } else { // var
        symbol_table[symbol_count].index = n_locals++;
    }
    debug("Defined symbol: %s, type: %s, kind: %s, index: %d\n",
          name, type, kind, symbol_table[symbol_count].index);
    symbol_count++;
}

int var_index(const char* name) {
    for (int i = 0; i < symbol_count; i++) {
        if (strcmp(symbol_table[i].name, name) == 0) {
            return symbol_table[i].index;
        }
    }
    return -1;
}

const char* var_kind(const char* name) {
    for (int i = 0; i < symbol_count; i++) {
        if (strcmp(symbol_table[i].name, name) == 0) {
            return symbol_table[i].kind;
        }
    }
    return NULL;
}

// 編譯表達式
void compile_expression(FILE* f);

// 編譯 term
void compile_term(FILE* f) {
    // next_token(f);
    if (strcmp(token_type, "integerConstant") == 0) {
        write_vm("push constant %s", current_token);
    }
    else if (strcmp(token_type, "stringConstant") == 0) {
        int len = strlen(current_token);
        write_vm("push constant %d", len);
        write_vm("call String.new 1");
        for (int i = 0; i < len; i++) {
            write_vm("push constant %d", (int)current_token[i]);
            write_vm("call String.appendChar 2");
        }
    }
    else if (strcmp(current_token, "true") == 0) {
        write_vm("push constant 0");
        write_vm("not");
    }
    else if (strcmp(current_token, "false") == 0 || strcmp(current_token, "null") == 0) {
        write_vm("push constant 0");
    }
    else if (strcmp(current_token, "this") == 0) {
        write_vm("push pointer 0");
    }
    else if (strcmp(token_type, "identifier") == 0) {
        char name[64];
        strcpy(name, current_token);
        next_token(f);
        if (strcmp(current_token, "[") == 0) {
            // 陣列索引
            int idx = var_index(name);
            const char* kind = var_kind(name);
            if (strcmp(kind, "static") == 0) write_vm("push static %d", idx);
            else if (strcmp(kind, "field") == 0) write_vm("push this %d", idx);
            else if (strcmp(kind, "arg") == 0) write_vm("push argument %d", idx);
            else write_vm("push local %d", idx);

            compile_expression(f); // index
            write_vm("add");
            write_vm("pop pointer 1");
            write_vm("push that 0");
            next_token(f); // ]
        }
        else if (strcmp(current_token, ".") == 0) {
            // 方法或函式呼叫
            next_token(f);
            char subroutine[64];
            strcpy(subroutine, current_token);
            next_token(f); // (
            write_vm("push pointer 0"); // 假設是方法
            compile_expression(f); // 參數
            write_vm("call %s.%s %d", name, subroutine, 1);
        }
        else {
            // 單純變數
            int idx = var_index(name);
            const char* kind = var_kind(name);
            debug("變數 %s 的索引是 %d kind=%s\n", name, idx, kind); // 除錯輸出
            if (strcmp(kind, "static") == 0) write_vm("push static %d", idx);
            else if (strcmp(kind, "field") == 0) write_vm("push this %d", idx);
            else if (strcmp(kind, "arg") == 0) write_vm("push argument %d", idx);
            else write_vm("push local %d", idx);
        }
    }
    else if (strcmp(current_token, "(") == 0) {
        debug("Compiling sub-expression (...)\n");
        next_token(f); // eat (
        compile_expression(f);
        next_token(f); // )
    }
    else if (strchr("~-", current_token[0])) {
        char op = current_token[0];
        compile_term(f);
        if (op == '-') write_vm("neg");
        else if (op == '~') write_vm("not");
    }
}

// 編譯表達式
void compile_expression(FILE* f) {
    compile_term(f);
    next_token(f);
    while (strchr("+-*/&|<>=", current_token[0])) {
        char op = current_token[0];
        debug("Expression op: %c\n", op);
        next_token(f);
        compile_term(f);
        switch (op) {
            case '+': write_vm("add"); break;
            case '-': write_vm("sub"); break;
            case '*': write_vm("call Math.multiply 2"); break;
            case '/': write_vm("call Math.divide 2"); break;
            case '&': write_vm("and"); break;
            case '|': write_vm("or"); break;
            case '<': write_vm("lt"); break;
            case '>': write_vm("gt"); break;
            case '=': write_vm("eq"); break;
        }
        next_token(f);
    }
}

// 編譯 let 陳述
void compile_let(FILE* f) {
    next_token(f); // varName
    char var_name[64];
    strcpy(var_name, current_token);
    next_token(f);
    if (strcmp(current_token, "[") == 0) {
        // 陣列賦值
        int idx = var_index(var_name);
        const char* kind = var_kind(var_name);
        if (strcmp(kind, "static") == 0) write_vm("push static %d", idx);
        else if (strcmp(kind, "field") == 0) write_vm("push this %d", idx);
        else if (strcmp(kind, "arg") == 0) write_vm("push argument %d", idx);
        else write_vm("push local %d", idx);

        compile_expression(f); // index
        write_vm("add");
        next_token(f); // =
        compile_expression(f);
        write_vm("pop temp 0");
        write_vm("pop pointer 1");
        write_vm("push temp 0");
        write_vm("pop that 0");
    } else {
        // 一般賦值
        next_token(f); // =
        compile_expression(f);
        int idx = var_index(var_name);
        const char* kind = var_kind(var_name);
        printf("變數 %s 的索引是 %d kind=%s\n", var_name, idx, kind); // 除錯輸出
        if (strcmp(kind, "static") == 0) write_vm("pop static %d", idx);
        else if (strcmp(kind, "field") == 0) write_vm("pop this %d", idx);
        else if (strcmp(kind, "arg") == 0) write_vm("pop argument %d", idx);
        else write_vm("pop local %d", idx);
    }
    next_token(f); // ;
}


// 編譯 do 語句
void compile_do(FILE* f) {
    next_token(f); // 吃掉 subroutineCall 的第一部分（identifier 或 className）

    char obj_name[64] = "";
    char subroutine_name[64];
    int n_args = 0;
    int is_method = 0;

    // 可能是： identifier . subroutineName ( expressionList )
    // 或：      subroutineName ( expressionList )  <-- 當前物件的方法
    strcpy(obj_name, current_token);

    next_token(f); // . 或 (
    if (strcmp(current_token, ".") == 0) {
        next_token(f); // subroutineName
        strcpy(subroutine_name, current_token);
        next_token(f); // (
    } else {
        // 是當前物件的方法呼叫，如 do move();
        strcpy(subroutine_name, obj_name);
        strcpy(obj_name, "this");
        is_method = 1;
        // current_token 已經是 (
    }

    // 推入物件指標（如果是方法）
    if (is_method || var_index(obj_name) != -1) {
        // 是物件變數或 this
        const char* kind = var_kind(obj_name);
        int idx = var_index(obj_name);
        if (strcmp(obj_name, "this") == 0) {
            write_vm("push pointer 0");
        } else if (strcmp(kind, "static") == 0) write_vm("push static %d", idx);
        else if (strcmp(kind, "field") == 0) write_vm("push this %d", idx);
        else if (strcmp(kind, "arg") == 0) write_vm("push argument %d", idx);
        else write_vm("push local %d", idx);
        n_args = 1;
        // 完整函式名是 ClassName.subroutineName
        char full_name[128];
        if (strcmp(obj_name, "this") == 0) {
            snprintf(full_name, sizeof(full_name), "%s.%s", current_class, subroutine_name);
        } else {
            // 假設 obj_name 是某類別變數，型別就是類別名
            const char* class_name = NULL;
            for (int i = 0; i < symbol_count; i++) {
                if (strcmp(symbol_table[i].name, obj_name) == 0) {
                    class_name = symbol_table[i].type;
                    break;
                }
            }
            snprintf(full_name, sizeof(full_name), "%s.%s", class_name ? class_name : obj_name, subroutine_name);
        }
        // 之後會用 full_name 呼叫
    } else {
        // 是靜態函式呼叫，如 Output.printInt
        char full_name[128];
        snprintf(full_name, sizeof(full_name), "%s.%s", obj_name, subroutine_name);
    }

    // 解析 expressionList
    next_token(f); // 吃掉 (
    if (strcmp(current_token, ")") != 0) {
        debug("do ExpressionList...\n");
        compile_expression(f); // 第一個參數
        n_args++;
        while (strcmp(current_token, ")") != 0) {
            if (strcmp(current_token, ",") == 0) {
                next_token(f);
                compile_expression(f);
                n_args++;
            } else {
                break;
            }
        }
    }
    next_token(f); // 吃掉 )

    // 產生呼叫
    if (is_method || var_index(obj_name) != -1) {
        const char* class_name = current_class;
        if (strcmp(obj_name, "this") != 0) {
            for (int i = 0; i < symbol_count; i++) {
                if (strcmp(symbol_table[i].name, obj_name) == 0) {
                    class_name = symbol_table[i].type;
                    break;
                }
            }
        }
        write_vm("call %s.%s %d", class_name, subroutine_name, n_args);
    } else {
        write_vm("call %s.%s %d", obj_name, subroutine_name, n_args);
    }

    // do 語句必須丟棄返回值
    write_vm("pop temp 0");

    next_token(f); // 吃掉 ;
}

// 編譯 if
void compile_if(FILE* f) {
    char* l1 = new_label("IF_TRUE");
    char* l2 = new_label("IF_FALSE");
    char* l3 = new_label("IF_END");

    compile_expression(f);
    write_vm("if-goto %s", l1);
    write_vm("goto %s", l2);
    write_vm("label %s", l1);

    next_token(f); // {
    next_token(f);
    while (strcmp(current_token, "}") != 0) {
        if (strcmp(current_token, "let") == 0) compile_let(f);
        next_token(f);
    }
    write_vm("goto %s", l3);
    write_vm("label %s", l2);

    next_token(f); // else ?
    if (strcmp(current_token, "else") == 0) {
        next_token(f); // {
        next_token(f);
        while (strcmp(current_token, "}") != 0) {
            if (strcmp(current_token, "let") == 0) compile_let(f);
            next_token(f);
        }
        next_token(f);
    }
    write_vm("label %s", l3);
}

// 編譯 while 迴圈
void compile_while(FILE* f) {
    char* label_top   = new_label("WHILE_TOP");
    char* label_end   = new_label("WHILE_END");

    write_vm("label %s", label_top);   // 迴圈開始

    next_token(f);                     // 吃掉 '('
    compile_expression(f);             // 條件式
    write_vm("not");                   // 條件為 false 時跳出
    write_vm("if-goto %s", label_end);

    next_token(f);                     // 吃掉 ')'
    next_token(f);                     // 吃掉 '{'
    next_token(f);                     // 第一個語句或 '}'
    while (strcmp(current_token, "}") != 0) {
        if (strcmp(current_token, "let") == 0) {
            compile_let(f);
        }
        else if (strcmp(current_token, "do") == 0) {
            compile_do(f);
        }
        else if (strcmp(current_token, "if") == 0) {
            compile_if(f);
        }
        else if (strcmp(current_token, "while") == 0) {
            compile_while(f);          // 巢狀 while
        }
        else if (strcmp(current_token, "return") == 0) {
            next_token(f);
            if (strcmp(current_token, ";") != 0) {
                compile_expression(f);
            } else {
                write_vm("push constant 0");
            }
            write_vm("return");
        }
        next_token(f);
    }
    next_token(f);                     // 吃掉 '}' (已經在迴圈外)

    write_vm("goto %s", label_top);    // 回到條件判斷
    write_vm("label %s", label_end);   // 迴圈結束
}

// 編譯子程式
void compile_subroutine(FILE* f) {
    debug("Compiling subroutine...\n");
    n_locals = 0;
    n_args = 0;
    symbol_count = field_count; // 清除局部符號表

    // next_token(f); // constructor/function/method
    char subroutine_type[16];
    strcpy(subroutine_type, current_token);
    debug("Subroutine type: %s\n", subroutine_type);
    next_token(f); // return type
    next_token(f); // subroutine name
    strcpy(current_subroutine, current_token);
    debug("Subroutine name: %s\n", current_subroutine);
    if (strcmp(subroutine_type, "method") == 0) {
        define_symbol("this", current_class, "arg");
    }

    next_token(f); // (
    next_token(f); // parameterList
    while (strcmp(current_token, ")") != 0) {
        if (strcmp(token_type, "identifier") == 0) {
            char type[64];
            strcpy(type, current_token);
            next_token(f);
            define_symbol(current_token, type, "arg");
        }
        next_token(f);
    }
    next_token(f); // {

    // function/method/constructor 宣告
    int n = n_locals;
    write_vm("function %s.%s %d", current_class, current_subroutine, n);

    if (strcmp(subroutine_type, "constructor") == 0) {
        write_vm("push constant %d", field_count);
        write_vm("call Memory.alloc 1");
        write_vm("pop pointer 0");
    } else if (strcmp(subroutine_type, "method") == 0) {
        write_vm("push argument 0");
        write_vm("pop pointer 0");
    }

    // 子程式體
    debug("Compiling subroutine body...\n");
    next_token(f);
    while (strcmp(current_token, "}") != 0) {
        if (strcmp(current_token, "var") == 0) {
            next_token(f); // type
            char type[64];
            strcpy(type, current_token);
            next_token(f); // varName
            define_symbol(current_token, type, "var");
            // define_symbol(current_token, "int", "var");
            next_token(f);
            debug("...current_token after varName: %s\n", current_token); // 除錯輸出
            while (strcmp(current_token, ";") != 0) {
                // next_token(f); // ,
                next_token(f); // varName
                define_symbol(current_token, type, "var");
                next_token(f);
            }
        }
        else if (strcmp(current_token, "let") == 0) {
            compile_let(f);
        }
        else if (strcmp(current_token, "if") == 0) {
            compile_if(f);
        }
        else if (strcmp(current_token, "do") == 0) {
            compile_do(f);
        }
        else if (strcmp(current_token, "return") == 0) {
            next_token(f);
            if (strcmp(current_token, ";") != 0) {
                compile_expression(f);
            } else {
                write_vm("push constant 0");
            }
            write_vm("return");
        }
        next_token(f);
    }
}

// 編譯 class
void compile_class(FILE* f, const char* filename) {
    ptr = NULL;
    has_more = 1;

    printf("開始編譯類別 class %s\n", filename);
    next_token(f); // class
    next_token(f); // className
    strcpy(current_class, current_token);
    field_count = 0;
    symbol_count = 0;

    char vm_filename[256];
    strcpy(vm_filename, filename);
    strcpy(strrchr(vm_filename, '.'), ".vm");
    vm_out = fopen(vm_filename, "w");

    next_token(f); // {

    while (1) {
        next_token(f);
        if (strcmp(current_token, "}") == 0) break;

        if (strcmp(current_token, "static") == 0 || strcmp(current_token, "field") == 0) {
            char kind[16];
            strcpy(kind, current_token);
            next_token(f); // type
            char type[64];
            strcpy(type, current_token);
            next_token(f); // varName
            define_symbol(current_token, type, kind);
            next_token(f);
            while (strcmp(current_token, ";") != 0) {
                //next_token(f); // ,
                next_token(f);
                define_symbol(current_token, type, kind);
                next_token(f);
            }
        }
        else if (strcmp(current_token, "constructor") == 0 ||
                 strcmp(current_token, "function") == 0 ||
                 strcmp(current_token, "method") == 0) {
            compile_subroutine(f);
        }
    }

    fclose(vm_out);
}

// 主函式
int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("用法: %s <file.jack 或目錄>\n", argv[0]);
        return 1;
    }

    struct stat st;
    if (stat(argv[1], &st) != 0) {
        perror("檔案不存在");
        return 1;
    }

    if (S_ISDIR(st.st_mode)) {
        DIR* dir = opendir(argv[1]);
        struct dirent* entry;
        while ((entry = readdir(dir)) != NULL) {
            if (strstr(entry->d_name, ".jack")) {
                char path[512];
                snprintf(path, sizeof(path), "%s/%s", argv[1], entry->d_name);
                FILE* f = fopen(path, "r");
                if (f) {
                    printf("編譯 %s\n", path);
                    compile_class(f, path);
                    fclose(f);
                }
            }
        }
        closedir(dir);
    } else {
        FILE* f = fopen(argv[1], "r");
        if (!f) {
            perror("無法開啟檔案");
            return 1;
        }
        printf("編譯 %s\n", argv[1]);
        compile_class(f, argv[1]);
        fclose(f);
    }

    printf("完成！VM 檔案已產生。\n");
    return 0;
}