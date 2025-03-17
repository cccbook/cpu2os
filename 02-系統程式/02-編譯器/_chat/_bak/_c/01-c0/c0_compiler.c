#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

// 堆疊機指令集
enum { OP_PUSH, OP_LOAD, OP_STORE, OP_JMP, OP_JIF, OP_HALT, OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_LT, OP_GT };

char *op_names[] = {
    "PUSH", "LOAD", "STORE", "JMP", "JIF", "HALT",  "ADD", "SUB", "MUL", "DIV", "LT", "GT"
};

// 最大程式長度和符號表大小
#define MAX_CODE_SIZE 256
#define MAX_SYMBOLS 10

// Token 類型
typedef enum {
    TOK_NUMBER, TOK_VARIABLE, TOK_OPERATOR, TOK_EQ, TOK_SEMI,
    TOK_IF, TOK_WHILE, TOK_LBRACE, TOK_RBRACE, TOK_LPAREN, TOK_RPAREN, TOK_EOF
} TokenType;

// Token 結構
typedef struct {
    TokenType type;
    int value;        // 數字的值
    char text[16];    // 變數名或運算符
} Token;

// 符號表結構
typedef struct {
    char name[16];
    int value;
    int id;
} Symbol;

typedef struct {
    Symbol symbols[MAX_SYMBOLS];
    int count;
} SymbolTable;

// 詞法分析器結構
typedef struct {
    const char* input;
    int pos;
} Lexer;

void compile_statement(Lexer* lexer, int* code, int* pos, SymbolTable* table);

// 初始化詞法分析器
void init_lexer(Lexer* lexer, const char* input) {
    lexer->input = input;
    lexer->pos = 0;
}

// 獲取下一個 token
Token next_token(Lexer* lexer) {
    Token token;
    while (isspace(lexer->input[lexer->pos])) lexer->pos++;

    if (lexer->input[lexer->pos] == '\0') {
        token.type = TOK_EOF;
        return token;
    }

    if (isdigit(lexer->input[lexer->pos])) {
        token.type = TOK_NUMBER;
        token.value = atoi(&lexer->input[lexer->pos]);
        while (isdigit(lexer->input[lexer->pos])) lexer->pos++;
        return token;
    }

    if (isalpha(lexer->input[lexer->pos])) {
        int i = 0;
        while (isalnum(lexer->input[lexer->pos]) && i < 15) {
            token.text[i++] = lexer->input[lexer->pos++];
        }
        token.text[i] = '\0';
        if (strcmp(token.text, "if") == 0) token.type = TOK_IF;
        else if (strcmp(token.text, "while") == 0) token.type = TOK_WHILE;
        else token.type = TOK_VARIABLE;
        return token;
    }

    token.text[0] = lexer->input[lexer->pos++];
    token.text[1] = '\0';
    if (token.text[0] == '=') token.type = TOK_EQ;
    else if (token.text[0] == ';') token.type = TOK_SEMI;
    else if (token.text[0] == '{') token.type = TOK_LBRACE;
    else if (token.text[0] == '}') token.type = TOK_RBRACE;
    else if (token.text[0] == '(') token.type = TOK_LPAREN;
    else if (token.text[0] == ')') token.type = TOK_RPAREN;
    else if (strchr("+-*/<>", token.text[0])) token.type = TOK_OPERATOR;
    else {
        printf("未知字符: %c\n", token.text[0]);
        exit(1);
    }
    return token;
}

// 前瞻 token
Token peek(Lexer* lexer) {
    int old_pos = lexer->pos;
    Token token = next_token(lexer);
    lexer->pos = old_pos;
    return token;
}

// 符號表操作
int add_symbol(SymbolTable* table, const char* name) {
    for (int i = 0; i < table->count; i++) {
        if (strcmp(table->symbols[i].name, name) == 0) return i;
    }
    if (table->count >= MAX_SYMBOLS) {
        printf("符號表已滿\n");
        exit(1);
    }
    strcpy(table->symbols[table->count].name, name);
    table->symbols[table->count].value = 0;
    table->symbols[table->count].id = table->count;
    return table->count++;
}

// 編譯 <expression>
void compile_expression(Lexer* lexer, int* code, int* pos, SymbolTable* table) {
    Token token = next_token(lexer);
    if (token.type == TOK_NUMBER) {
        code[(*pos)++] = OP_PUSH;
        code[(*pos)++] = (int)token.value;
    } else if (token.type == TOK_VARIABLE) {
        int id = add_symbol(table, token.text);
        code[(*pos)++] = OP_LOAD;
        code[(*pos)++] = (int)id;
    } else {
        printf("預期數字或變數\n");
        exit(1);
    }

    Token next = peek(lexer);
    if (next.type == TOK_OPERATOR) {
        token = next_token(lexer); // 吃掉運算符
        compile_expression(lexer, code, pos, table);
        switch (token.text[0]) {
            case '+': code[(*pos)++] = OP_ADD; break;
            case '-': code[(*pos)++] = OP_SUB; break;
            case '*': code[(*pos)++] = OP_MUL; break;
            case '/': code[(*pos)++] = OP_DIV; break;
            case '<': code[(*pos)++] = OP_LT; break;
            case '>': code[(*pos)++] = OP_GT; break;
        }
    }
}

// 編譯 <assignment>
void compile_assignment(Lexer* lexer, int* code, int* pos, SymbolTable* table) {
    Token token = next_token(lexer);
    if (token.type != TOK_VARIABLE) {
        printf("賦值需以變數開始\n");
        exit(1);
    }
    int id = add_symbol(table, token.text);

    token = next_token(lexer);
    if (token.type != TOK_EQ) {
        printf("預期 '='\n");
        exit(1);
    }

    compile_expression(lexer, code, pos, table);
    code[(*pos)++] = OP_STORE;
    code[(*pos)++] = (int)id;

    token = next_token(lexer);
    if (token.type != TOK_SEMI) {
        printf("預期 ';'\n");
        exit(1);
    }
}

// 編譯 <block>
void compile_block(Lexer* lexer, int* code, int* pos, SymbolTable* table) {
    Token token = next_token(lexer);
    if (token.type != TOK_LBRACE) {
        printf("預期 '{'\n");
        exit(1);
    }

    while (peek(lexer).type != TOK_RBRACE) {
        compile_statement(lexer, code, pos, table);
    }
    next_token(lexer); // 吃掉 "}"
}

// 編譯 <statement>
void compile_statement(Lexer* lexer, int* code, int* pos, SymbolTable* table) {
    Token token = peek(lexer);
    if (token.type == TOK_VARIABLE) {
        compile_assignment(lexer, code, pos, table);
    } else if (token.type == TOK_IF) {
        next_token(lexer); // 吃掉 "if"
        token = next_token(lexer);
        if (token.type != TOK_LPAREN) {
            printf("預期 '('\n");
            exit(1);
        }

        compile_expression(lexer, code, pos, table);
        token = next_token(lexer);
        if (token.type != TOK_RPAREN) {
            printf("預期 ')'\n");
            exit(1);
        }

        int jif_pos = *pos;
        code[(*pos)++] = OP_JIF;
        code[(*pos)++] = 0; // 占位符

        compile_block(lexer, code, pos, table);
        code[jif_pos + 1] = (int)(*pos - jif_pos - 2); // 更新跳轉偏移
    } else if (token.type == TOK_WHILE) {
        next_token(lexer); // 吃掉 "while"
        token = next_token(lexer);
        if (token.type != TOK_LPAREN) {
            printf("預期 '('\n");
            exit(1);
        }

        int loop_start = *pos;
        compile_expression(lexer, code, pos, table);
        token = next_token(lexer);
        if (token.type != TOK_RPAREN) {
            printf("預期 ')'\n");
            exit(1);
        }

        int jif_pos = *pos;
        code[(*pos)++] = OP_JIF;
        code[(*pos)++] = 0; // 占位符

        compile_block(lexer, code, pos, table);
        code[(*pos)++] = OP_JMP;
        code[(*pos)++] = (int)(loop_start - *pos - 1); // 跳回開頭
        code[jif_pos + 1] = (int)(*pos - jif_pos - 2); // 更新跳出偏移
    } else {
        printf("未知語句\n");
        exit(1);
    }
}

// 編譯 <program>
void compile_program(Lexer* lexer, int* code, int* pos, SymbolTable* table) {
    while (peek(lexer).type != TOK_EOF) {
        compile_statement(lexer, code, pos, table);
    }
    code[(*pos)++] = OP_HALT;
}

// 執行堆疊機
void run(int* code, int size, SymbolTable* table) {
    int stack[256], sp = -1, pc = 0;
    while (pc < size) {
        printf("%d:%s ", pc, op_names[code[pc]]);
        if (code[pc] < OP_HALT) printf("%d", code[pc+1]);
        printf("\n");
        switch (code[pc++]) {
            case OP_PUSH: stack[++sp] = code[pc++]; break;
            case OP_LOAD: stack[++sp] = table->symbols[code[pc++]].value; break;
            case OP_STORE: table->symbols[code[pc++]].value = stack[sp--]; break;
            case OP_ADD: stack[sp-1] += stack[sp]; sp--; break;
            case OP_SUB: stack[sp-1] -= stack[sp]; sp--; break;
            case OP_MUL: stack[sp-1] *= stack[sp]; sp--; break;
            case OP_DIV: stack[sp-1] /= stack[sp]; sp--; break;
            case OP_LT: stack[sp-1] = (stack[sp-1]<stack[sp]); sp--; break;
            case OP_GT: stack[sp-1] = (stack[sp-1]>stack[sp]); sp--; break;
            case OP_JMP: pc += code[pc]; break;
            case OP_JIF: if (!stack[sp--]) pc += code[pc]; else pc++; break;
            case OP_HALT: printf("執行結束，%s = %d\n", table->symbols[0].name, table->symbols[0].value); return;
        }
    }
}

// 主函數
int main() {
    // const char* input = "x = 5; while (x - 0) { x = x - 1; }";
    const char* input = "sum=0; i=1; while (i<10) { i = i + 1; sum = sum + i; }";
    printf("編譯並執行程式: %s\n", input);

    SymbolTable table = { .count = 0 };
    Lexer lexer;
    init_lexer(&lexer, input);

    int code[MAX_CODE_SIZE];
    int pos = 0;
    compile_program(&lexer, code, &pos, &table);

    // 輸出生成的指令
    printf("生成的堆疊機指令:\n");
    for (int i = 0; i < pos;) {
        switch (code[i]) {
            case OP_PUSH: printf("PUSH %d\n", code[i+1]); i += 2; break;
            case OP_LOAD: printf("LOAD %d\n", code[i+1]); i += 2; break;
            case OP_STORE: printf("STORE %d\n", code[i+1]); i += 2; break;
            case OP_ADD: printf("ADD\n"); i++; break;
            case OP_SUB: printf("SUB\n"); i++; break;
            case OP_MUL: printf("MUL\n"); i++; break;
            case OP_DIV: printf("DIV\n"); i++; break;
            case OP_LT: printf("LT\n"); i++; break;
            case OP_GT: printf("GT\n"); i++; break;
            case OP_JMP: printf("JMP %d\n", code[i+1]); i += 2; break;
            case OP_JIF: printf("JIF %d\n", code[i+1]); i += 2; break;
            case OP_HALT: printf("HALT\n"); i++; break;
        }
    }

    // 執行
    run(code, pos, &table);
    return 0;
}