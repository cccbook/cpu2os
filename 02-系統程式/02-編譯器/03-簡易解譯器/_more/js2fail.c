#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

double parse_expression();
void parse_statement();

// 定義 token 類型（保持不變）
typedef enum {
    TOKEN_NUMBER,
    TOKEN_IDENTIFIER,
    TOKEN_PLUS,
    TOKEN_MINUS,
    TOKEN_MULTIPLY,
    TOKEN_DIVIDE,
    TOKEN_ASSIGN,
    TOKEN_SEMICOLON,
    TOKEN_LPAREN,
    TOKEN_RPAREN,
    TOKEN_IF,
    TOKEN_WHILE,
    TOKEN_LBRACE,    // 新增：左大括號
    TOKEN_RBRACE,    // 新增：右大括號
    TOKEN_END,
    TOKEN_ERROR
} TokenType;

// token 結構（保持不變）
typedef struct {
    TokenType type;
    char* value;
    double num_value;
} Token;

// 變數結構（保持不變）
typedef struct {
    char* name;
    double value;
} Variable;

#define MAX_TOKENS 1000
#define MAX_VARS 100

// 全局變數（保持不變）
Token tokens[MAX_TOKENS];
int token_count = 0;
Variable variables[MAX_VARS];
int var_count = 0;
int current_token = 0;

// 詞法分析器（添加大括號支持）
void tokenize(const char* source) {
    int pos = 0;
    token_count = 0;
    
    while (source[pos] != '\0') {
        while (isspace(source[pos])) pos++;
        if (source[pos] == '\0') break;
        
        Token* token = &tokens[token_count++];
        
        if (isdigit(source[pos])) {
            char buffer[32];
            int i = 0;
            while (isdigit(source[pos]) || source[pos] == '.') {
                buffer[i++] = source[pos++];
            }
            buffer[i] = '\0';
            token->type = TOKEN_NUMBER;
            token->num_value = atof(buffer);
            token->value = strdup(buffer);
            continue;
        }
        
        if (isalpha(source[pos])) {
            char buffer[32];
            int i = 0;
            while (isalnum(source[pos])) {
                buffer[i++] = source[pos++];
            }
            buffer[i] = '\0';
            token->value = strdup(buffer);
            if (strcmp(buffer, "if") == 0) token->type = TOKEN_IF;
            else if (strcmp(buffer, "while") == 0) token->type = TOKEN_WHILE;
            else token->type = TOKEN_IDENTIFIER;
            continue;
        }
        
        switch (source[pos]) {
            case '+': token->type = TOKEN_PLUS; token->value = strdup("+"); pos++; break;
            case '-': token->type = TOKEN_MINUS; token->value = strdup("-"); pos++; break;
            case '*': token->type = TOKEN_MULTIPLY; token->value = strdup("*"); pos++; break;
            case '/': token->type = TOKEN_DIVIDE; token->value = strdup("/"); pos++; break;
            case '=': token->type = TOKEN_ASSIGN; token->value = strdup("="); pos++; break;
            case ';': token->type = TOKEN_SEMICOLON; token->value = strdup(";"); pos++; break;
            case '(': token->type = TOKEN_LPAREN; token->value = strdup("("); pos++; break;
            case ')': token->type = TOKEN_RPAREN; token->value = strdup(")"); pos++; break;
            case '{': token->type = TOKEN_LBRACE; token->value = strdup("{"); pos++; break;
            case '}': token->type = TOKEN_RBRACE; token->value = strdup("}"); pos++; break;
            default: token->type = TOKEN_ERROR; token->value = strdup("error"); pos++; break;
        }
    }
    
    tokens[token_count].type = TOKEN_END;
}

// 變數管理函數（保持不變）
double get_variable(const char* name) {
    for (int i = 0; i < var_count; i++) {
        if (strcmp(variables[i].name, name) == 0) {
            return variables[i].value;
        }
    }
    return 0;
}

void set_variable(const char* name, double value) {
    for (int i = 0; i < var_count; i++) {
        if (strcmp(variables[i].name, name) == 0) {
            variables[i].value = value;
            return;
        }
    }
    variables[var_count].name = strdup(name);
    variables[var_count].value = value;
    var_count++;
}

// token 操作函數（保持不變）
Token* peek() {
    return &tokens[current_token];
}

Token* consume() {
    return &tokens[current_token++];
}


// 表達式解析函數（保持不變）
double parse_factor() {
    Token* token = consume();
    if (token->type == TOKEN_NUMBER) {
        return token->num_value;
    }
    else if (token->type == TOKEN_IDENTIFIER) {
        return get_variable(token->value);
    }
    else if (token->type == TOKEN_LPAREN) {
        double result = parse_expression();
        consume(); // 消費右括號
        return result;
    }
    return 0;
}

double parse_term() {
    double left = parse_factor();
    while (peek()->type == TOKEN_MULTIPLY || peek()->type == TOKEN_DIVIDE) {
        Token* op = consume();
        double right = parse_factor();
        if (op->type == TOKEN_MULTIPLY) left *= right;
        else if (op->type == TOKEN_DIVIDE) left /= right;
    }
    return left;
}

double parse_expression() {
    double left = parse_term();
    while (peek()->type == TOKEN_PLUS || peek()->type == TOKEN_MINUS) {
        Token* op = consume();
        double right = parse_term();
        if (op->type == TOKEN_PLUS) left += right;
        else if (op->type == TOKEN_MINUS) left -= right;
    }
    return left;
}

// 解析語句塊
void parse_block() {
    consume(); // 消費左大括號
    while (peek()->type != TOKEN_RBRACE && peek()->type != TOKEN_END) {
        parse_statement();
    }
    consume(); // 消費右大括號
}

// 更新後的語句解析，支持 while 迴圈
void parse_statement() {
    Token* token = peek();
    
    if (token->type == TOKEN_IDENTIFIER) {
        char* var_name = consume()->value;
        if (peek()->type == TOKEN_ASSIGN) {
            consume(); // 消費等號
            double value = parse_expression();
            set_variable(var_name, value);
            consume(); // 消費分號
        }
    }
    else if (token->type == TOKEN_IF) {
        consume(); // 消費 if
        consume(); // 消費左括號
        double condition = parse_expression();
        consume(); // 消費右括號
        if (condition != 0) {
            if (peek()->type == TOKEN_LBRACE) {
                parse_block();
            } else {
                parse_statement();
            }
        }
    }
    else if (token->type == TOKEN_WHILE) {
        consume(); // 消費 while
        int condition_pos = current_token;
        
        while (1) {
            consume(); // 消費左括號
            double condition = parse_expression();
            consume(); // 消費右括號
            
            if (condition == 0) break;
            
            if (peek()->type == TOKEN_LBRACE) {
                parse_block();
            } else {
                parse_statement();
            }
            current_token = condition_pos; // 回到條件判斷位置
        }
    }
}

// 主解析函數（保持不變）
void parse(const char* source) {
    tokenize(source);
    current_token = 0;
    while (peek()->type != TOKEN_END) {
        parse_statement();
    }
}

// 主函數，展示 while 迴圈使用
int main() {
    const char* code = "x = 5; while (x) { x = x - 1; y = y + 2; }";
    parse(code);
    
    printf("x = %.2f\n", get_variable("x"));  // 應輸出 0
    printf("y = %.2f\n", get_variable("y"));  // 應輸出 10
    
    return 0;
}