#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdbool.h>

#define MAX_TOKEN_LEN 256
#define MAX_SYMBOLS 1000
#define MAX_LINE 1024

// Token 類型
typedef enum {
    TK_KEYWORD, TK_SYMBOL, TK_IDENTIFIER, 
    TK_INT_CONST, TK_STRING_CONST, TK_EOF
} TokenType;

// 關鍵字
typedef enum {
    KW_CLASS, KW_CONSTRUCTOR, KW_FUNCTION, KW_METHOD,
    KW_FIELD, KW_STATIC, KW_VAR, KW_INT, KW_CHAR,
    KW_BOOLEAN, KW_VOID, KW_TRUE, KW_FALSE, KW_NULL,
    KW_THIS, KW_LET, KW_DO, KW_IF, KW_ELSE,
    KW_WHILE, KW_RETURN
} Keyword;

// 符號表條目
typedef enum { SK_STATIC, SK_FIELD, SK_ARG, SK_VAR } SymbolKind;

typedef struct {
    char name[MAX_TOKEN_LEN];
    char type[MAX_TOKEN_LEN];
    SymbolKind kind;
    int index;
} Symbol;

// 符號表
typedef struct {
    Symbol symbols[MAX_SYMBOLS];
    int count;
} SymbolTable;

// 編譯器狀態
typedef struct {
    char *source;
    int pos;
    char current_token[MAX_TOKEN_LEN];
    TokenType token_type;
    FILE *output;
    SymbolTable class_table;
    SymbolTable subroutine_table;
    char class_name[MAX_TOKEN_LEN];
    int label_count;
} Compiler;

// 函數聲明
void advance(Compiler *c);
void compile_class(Compiler *c);
void compile_subroutine(Compiler *c);
void compile_statements(Compiler *c);
void compile_expression(Compiler *c);
void compile_term(Compiler *c);

// 初始化符號表
void init_symbol_table(SymbolTable *table) {
    table->count = 0;
}

// 添加符號
void add_symbol(SymbolTable *table, const char *name, const char *type, SymbolKind kind) {
    int idx = 0;
    for (int i = 0; i < table->count; i++) {
        if (table->symbols[i].kind == kind) idx++;
    }
    strcpy(table->symbols[table->count].name, name);
    strcpy(table->symbols[table->count].type, type);
    table->symbols[table->count].kind = kind;
    table->symbols[table->count].index = idx;
    table->count++;
}

// 查找符號
Symbol* find_symbol(Compiler *c, const char *name) {
    for (int i = 0; i < c->subroutine_table.count; i++) {
        if (strcmp(c->subroutine_table.symbols[i].name, name) == 0)
            return &c->subroutine_table.symbols[i];
    }
    for (int i = 0; i < c->class_table.count; i++) {
        if (strcmp(c->class_table.symbols[i].name, name) == 0)
            return &c->class_table.symbols[i];
    }
    return NULL;
}

// 跳過空白和註釋
void skip_whitespace(Compiler *c) {
    while (c->source[c->pos]) {
        if (isspace(c->source[c->pos])) {
            c->pos++;
        } else if (c->source[c->pos] == '/' && c->source[c->pos + 1] == '/') {
            while (c->source[c->pos] && c->source[c->pos] != '\n') c->pos++;
        } else if (c->source[c->pos] == '/' && c->source[c->pos + 1] == '*') {
            c->pos += 2;
            while (c->source[c->pos] && !(c->source[c->pos] == '*' && c->source[c->pos + 1] == '/'))
                c->pos++;
            if (c->source[c->pos]) c->pos += 2;
        } else {
            break;
        }
    }
}

// 讀取下一個 token
void advance(Compiler *c) {
    skip_whitespace(c);
    
    if (!c->source[c->pos]) {
        c->token_type = TK_EOF;
        return;
    }
    
    // 字符串常量
    if (c->source[c->pos] == '"') {
        c->pos++;
        int i = 0;
        while (c->source[c->pos] && c->source[c->pos] != '"') {
            c->current_token[i++] = c->source[c->pos++];
        }
        c->current_token[i] = '\0';
        if (c->source[c->pos]) c->pos++;
        c->token_type = TK_STRING_CONST;
        return;
    }
    
    // 符號
    if (strchr("{}()[].,;+-*/&|<>=~", c->source[c->pos])) {
        c->current_token[0] = c->source[c->pos++];
        c->current_token[1] = '\0';
        c->token_type = TK_SYMBOL;
        return;
    }
    
    // 數字
    if (isdigit(c->source[c->pos])) {
        int i = 0;
        while (isdigit(c->source[c->pos])) {
            c->current_token[i++] = c->source[c->pos++];
        }
        c->current_token[i] = '\0';
        c->token_type = TK_INT_CONST;
        return;
    }
    
    // 識別符或關鍵字
    if (isalpha(c->source[c->pos]) || c->source[c->pos] == '_') {
        int i = 0;
        while (isalnum(c->source[c->pos]) || c->source[c->pos] == '_') {
            c->current_token[i++] = c->source[c->pos++];
        }
        c->current_token[i] = '\0';
        
        // 檢查是否為關鍵字
        const char *keywords[] = {
            "class", "constructor", "function", "method", "field", "static",
            "var", "int", "char", "boolean", "void", "true", "false", "null",
            "this", "let", "do", "if", "else", "while", "return"
        };
        for (int j = 0; j < 21; j++) {
            if (strcmp(c->current_token, keywords[j]) == 0) {
                c->token_type = TK_KEYWORD;
                return;
            }
        }
        c->token_type = TK_IDENTIFIER;
        return;
    }
}

// 生成唯一標籤
void gen_label(Compiler *c, char *buf, const char *prefix) {
    sprintf(buf, "%s%d", prefix, c->label_count++);
}

// VM 段名稱
const char* segment_name(SymbolKind kind) {
    switch (kind) {
        case SK_STATIC: return "static";
        case SK_FIELD: return "this";
        case SK_ARG: return "argument";
        case SK_VAR: return "local";
        default: return "temp";
    }
}

// 編譯 class
void compile_class(Compiler *c) {
    advance(c); // class
    advance(c); // className
    strcpy(c->class_name, c->current_token);
    advance(c); // {
    
    // classVarDec*
    while (strcmp(c->current_token, "static") == 0 || strcmp(c->current_token, "field") == 0) {
        SymbolKind kind = strcmp(c->current_token, "static") == 0 ? SK_STATIC : SK_FIELD;
        advance(c); // static | field
        char type[MAX_TOKEN_LEN];
        strcpy(type, c->current_token);
        advance(c); // type
        
        do {
            if (strcmp(c->current_token, ",") == 0) advance(c);
            char name[MAX_TOKEN_LEN];
            strcpy(name, c->current_token);
            add_symbol(&c->class_table, name, type, kind);
            advance(c); // varName
        } while (strcmp(c->current_token, ";") != 0);
        advance(c); // ;
    }
    
    // subroutineDec*
    while (strcmp(c->current_token, "}") != 0) {
        compile_subroutine(c);
    }
}

// 編譯子程序
void compile_subroutine(Compiler *c) {
    init_symbol_table(&c->subroutine_table);
    
    char sub_type[MAX_TOKEN_LEN];
    strcpy(sub_type, c->current_token); // constructor | function | method
    advance(c);
    
    if (strcmp(sub_type, "method") == 0) {
        add_symbol(&c->subroutine_table, "this", c->class_name, SK_ARG);
    }
    
    advance(c); // return type
    char sub_name[MAX_TOKEN_LEN];
    strcpy(sub_name, c->current_token);
    advance(c); // subroutineName
    advance(c); // (
    
    // parameterList
    if (strcmp(c->current_token, ")") != 0) {
        do {
            if (strcmp(c->current_token, ",") == 0) advance(c);
            char type[MAX_TOKEN_LEN], name[MAX_TOKEN_LEN];
            strcpy(type, c->current_token);
            advance(c);
            strcpy(name, c->current_token);
            add_symbol(&c->subroutine_table, name, type, SK_ARG);
            advance(c);
        } while (strcmp(c->current_token, ")") != 0);
    }
    advance(c); // )
    advance(c); // {
    
    // varDec*
    int n_locals = 0;
    while (strcmp(c->current_token, "var") == 0) {
        advance(c);
        char type[MAX_TOKEN_LEN];
        strcpy(type, c->current_token);
        advance(c);
        
        do {
            if (strcmp(c->current_token, ",") == 0) advance(c);
            char name[MAX_TOKEN_LEN];
            strcpy(name, c->current_token);
            add_symbol(&c->subroutine_table, name, type, SK_VAR);
            n_locals++;
            advance(c);
        } while (strcmp(c->current_token, ";") != 0);
        advance(c);
    }
    
    // 輸出 function 聲明
    fprintf(c->output, "function %s.%s %d\n", c->class_name, sub_name, n_locals);
    
    // constructor: 分配記憶體
    if (strcmp(sub_type, "constructor") == 0) {
        int n_fields = 0;
        for (int i = 0; i < c->class_table.count; i++) {
            if (c->class_table.symbols[i].kind == SK_FIELD) n_fields++;
        }
        fprintf(c->output, "push constant %d\n", n_fields);
        fprintf(c->output, "call Memory.alloc 1\n");
        fprintf(c->output, "pop pointer 0\n");
    }
    
    // method: 設置 this
    if (strcmp(sub_type, "method") == 0) {
        fprintf(c->output, "push argument 0\n");
        fprintf(c->output, "pop pointer 0\n");
    }
    
    compile_statements(c);
    advance(c); // }
}

// 編譯語句
void compile_statements(Compiler *c) {
    while (strcmp(c->current_token, "}") != 0) {
        if (strcmp(c->current_token, "let") == 0) {
            advance(c);
            char var_name[MAX_TOKEN_LEN];
            strcpy(var_name, c->current_token);
            Symbol *sym = find_symbol(c, var_name);
            advance(c);
            
            bool is_array = strcmp(c->current_token, "[") == 0;
            if (is_array) {
                advance(c); // [
                compile_expression(c);
                advance(c); // ]
                fprintf(c->output, "push %s %d\n", segment_name(sym->kind), sym->index);
                fprintf(c->output, "add\n");
            }
            
            advance(c); // =
            compile_expression(c);
            advance(c); // ;
            
            if (is_array) {
                fprintf(c->output, "pop temp 0\n");
                fprintf(c->output, "pop pointer 1\n");
                fprintf(c->output, "push temp 0\n");
                fprintf(c->output, "pop that 0\n");
            } else {
                fprintf(c->output, "pop %s %d\n", segment_name(sym->kind), sym->index);
            }
        } else if (strcmp(c->current_token, "if") == 0) {
            char label1[50], label2[50];
            gen_label(c, label1, "IF_TRUE");
            gen_label(c, label2, "IF_FALSE");
            
            advance(c); // if
            advance(c); // (
            compile_expression(c);
            advance(c); // )
            fprintf(c->output, "if-goto %s\n", label1);
            fprintf(c->output, "goto %s\n", label2);
            fprintf(c->output, "label %s\n", label1);
            advance(c); // {
            compile_statements(c);
            advance(c); // }
            
            if (strcmp(c->current_token, "else") == 0) {
                char label3[50];
                gen_label(c, label3, "IF_END");
                fprintf(c->output, "goto %s\n", label3);
                fprintf(c->output, "label %s\n", label2);
                advance(c); // else
                advance(c); // {
                compile_statements(c);
                advance(c); // }
                fprintf(c->output, "label %s\n", label3);
            } else {
                fprintf(c->output, "label %s\n", label2);
            }
        } else if (strcmp(c->current_token, "while") == 0) {
            char label1[50], label2[50];
            gen_label(c, label1, "WHILE_EXP");
            gen_label(c, label2, "WHILE_END");
            
            fprintf(c->output, "label %s\n", label1);
            advance(c); // while
            advance(c); // (
            compile_expression(c);
            advance(c); // )
            fprintf(c->output, "not\n");
            fprintf(c->output, "if-goto %s\n", label2);
            advance(c); // {
            compile_statements(c);
            advance(c); // }
            fprintf(c->output, "goto %s\n", label1);
            fprintf(c->output, "label %s\n", label2);
        } else if (strcmp(c->current_token, "do") == 0) {
            advance(c); // do
            compile_term(c); // subroutineCall
            advance(c); // ;
            fprintf(c->output, "pop temp 0\n"); // 丟棄返回值
        } else if (strcmp(c->current_token, "return") == 0) {
            advance(c);
            if (strcmp(c->current_token, ";") != 0) {
                compile_expression(c);
            } else {
                fprintf(c->output, "push constant 0\n");
            }
            advance(c); // ;
            fprintf(c->output, "return\n");
        }
    }
}

// 編譯表達式
void compile_expression(Compiler *c) {
    compile_term(c);
    
    while (strchr("+-*/&|<>=", c->current_token[0]) && strlen(c->current_token) == 1) {
        char op = c->current_token[0];
        advance(c);
        compile_term(c);
        
        switch (op) {
            case '+': fprintf(c->output, "add\n"); break;
            case '-': fprintf(c->output, "sub\n"); break;
            case '*': fprintf(c->output, "call Math.multiply 2\n"); break;
            case '/': fprintf(c->output, "call Math.divide 2\n"); break;
            case '&': fprintf(c->output, "and\n"); break;
            case '|': fprintf(c->output, "or\n"); break;
            case '<': fprintf(c->output, "lt\n"); break;
            case '>': fprintf(c->output, "gt\n"); break;
            case '=': fprintf(c->output, "eq\n"); break;
        }
    }
}

// 編譯項
void compile_term(Compiler *c) {
    if (c->token_type == TK_INT_CONST) {
        fprintf(c->output, "push constant %s\n", c->current_token);
        advance(c);
    } else if (c->token_type == TK_STRING_CONST) {
        int len = strlen(c->current_token);
        fprintf(c->output, "push constant %d\n", len);
        fprintf(c->output, "call String.new 1\n");
        for (int i = 0; i < len; i++) {
            fprintf(c->output, "push constant %d\n", c->current_token[i]);
            fprintf(c->output, "call String.appendChar 2\n");
        }
        advance(c);
    } else if (strcmp(c->current_token, "true") == 0) {
        fprintf(c->output, "push constant 0\n");
        fprintf(c->output, "not\n");
        advance(c);
    } else if (strcmp(c->current_token, "false") == 0 || strcmp(c->current_token, "null") == 0) {
        fprintf(c->output, "push constant 0\n");
        advance(c);
    } else if (strcmp(c->current_token, "this") == 0) {
        fprintf(c->output, "push pointer 0\n");
        advance(c);
    } else if (strcmp(c->current_token, "(") == 0) {
        advance(c);
        compile_expression(c);
        advance(c); // )
    } else if (strchr("-~", c->current_token[0])) {
        char op = c->current_token[0];
        advance(c);
        compile_term(c);
        fprintf(c->output, op == '-' ? "neg\n" : "not\n");
    } else {
        char name[MAX_TOKEN_LEN];
        strcpy(name, c->current_token);
        advance(c);
        
        if (strcmp(c->current_token, "[") == 0) {
            Symbol *sym = find_symbol(c, name);
            advance(c);
            compile_expression(c);
            advance(c); // ]
            fprintf(c->output, "push %s %d\n", segment_name(sym->kind), sym->index);
            fprintf(c->output, "add\n");
            fprintf(c->output, "pop pointer 1\n");
            fprintf(c->output, "push that 0\n");
        } else if (strcmp(c->current_token, "(") == 0 || strcmp(c->current_token, ".") == 0) {
            int n_args = 0;
            char full_name[MAX_TOKEN_LEN * 2];
            
            if (strcmp(c->current_token, ".") == 0) {
                advance(c); // .
                Symbol *sym = find_symbol(c, name);
                if (sym) {
                    fprintf(c->output, "push %s %d\n", segment_name(sym->kind), sym->index);
                    sprintf(full_name, "%s.%s", sym->type, c->current_token);
                    n_args = 1;
                } else {
                    sprintf(full_name, "%s.%s", name, c->current_token);
                }
                advance(c);
            } else {
                fprintf(c->output, "push pointer 0\n");
                sprintf(full_name, "%s.%s", c->class_name, name);
                n_args = 1;
            }
            
            advance(c); // (
            if (strcmp(c->current_token, ")") != 0) {
                compile_expression(c);
                n_args++;
                while (strcmp(c->current_token, ",") == 0) {
                    advance(c);
                    compile_expression(c);
                    n_args++;
                }
            }
            advance(c); // )
            fprintf(c->output, "call %s %d\n", full_name, n_args);
        } else {
            Symbol *sym = find_symbol(c, name);
            if (sym) {
                fprintf(c->output, "push %s %d\n", segment_name(sym->kind), sym->index);
            }
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("使用方法: %s <input.jack> <output.vm>\n", argv[0]);
        return 1;
    }
    
    FILE *input = fopen(argv[1], "r");
    if (!input) {
        printf("無法開啟輸入檔案\n");
        return 1;
    }
    
    fseek(input, 0, SEEK_END);
    long size = ftell(input);
    fseek(input, 0, SEEK_SET);
    
    char *source = malloc(size + 1);
    fread(source, 1, size, input);
    source[size] = '\0';
    fclose(input);
    
    FILE *output = fopen(argv[2], "w");
    if (!output) {
        printf("無法開啟輸出檔案\n");
        free(source);
        return 1;
    }
    
    Compiler compiler = {
        .source = source,
        .pos = 0,
        .output = output,
        .label_count = 0
    };
    
    init_symbol_table(&compiler.class_table);
    init_symbol_table(&compiler.subroutine_table);
    
    advance(&compiler);
    compile_class(&compiler);
    
    fclose(output);
    free(source);
    
    printf("編譯完成！\n");
    return 0;
}