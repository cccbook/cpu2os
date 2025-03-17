#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_TOKEN_LEN 100
#define MAX_VARS 100
#define MAX_FUNCS 100
#define MAX_PARAMS 10

// Token types
typedef enum {
    TOKEN_EOF,
    TOKEN_NUMBER,
    TOKEN_VARIABLE,
    TOKEN_OPERATOR,
    TOKEN_SEMICOLON,
    TOKEN_IF,
    TOKEN_WHILE,
    TOKEN_LPAREN,
    TOKEN_RPAREN,
    TOKEN_LBRACE,
    TOKEN_RBRACE,
    TOKEN_EQUALS,
    TOKEN_COMPARISON,
    TOKEN_FUNCTION,    // 新增: function 關鍵字
    TOKEN_RETURN,      // 新增: return 關鍵字
    TOKEN_COMMA        // 新增: 逗號用於參數分隔
} TokenType;

// Token structure
typedef struct {
    TokenType type;
    char value[MAX_TOKEN_LEN];
} Token;

// Function parameter structure
typedef struct {
    char name[MAX_TOKEN_LEN];
    int value;
} Parameter;

// Function structure
typedef struct {
    char name[MAX_TOKEN_LEN];
    char *code;                    // 函數體的代碼
    Parameter params[MAX_PARAMS];  // 參數列表
    int param_count;               // 參數數量
    int start_pos;                 // 函數開始位置
} Function;

// Variable storage
typedef struct {
    char name[MAX_TOKEN_LEN];
    int value;
} Variable;

// Global variables
char *input;
int pos = 0;
Token current_token;
Variable variables[MAX_VARS];
int var_count = 0;
Function functions[MAX_FUNCS];  // 函數表
int func_count = 0;

// Function declarations
void get_next_token();
int parse_program();
int parse_statement();
int parse_block();
int parse_assignment();
int parse_expression();
int parse_function_definition();
int parse_function_call(const char *func_name);
int find_or_create_variable(const char *name);
int find_function(const char *name);

// Get next token from input
void get_next_token() {
    while (isspace(input[pos])) pos++;
    
    if (input[pos] == '\0') {
        current_token.type = TOKEN_EOF;
        return;
    }
    
    if (isdigit(input[pos])) {
        int i = 0;
        while (isdigit(input[pos])) {
            current_token.value[i++] = input[pos++];
        }
        current_token.value[i] = '\0';
        current_token.type = TOKEN_NUMBER;
        return;
    }
    
    if (isalpha(input[pos])) {
        int i = 0;
        while (isalnum(input[pos])) {
            current_token.value[i++] = input[pos++];
        }
        current_token.value[i] = '\0';
        
        if (strcmp(current_token.value, "if") == 0) {
            current_token.type = TOKEN_IF;
        } else if (strcmp(current_token.value, "while") == 0) {
            current_token.type = TOKEN_WHILE;
        } else if (strcmp(current_token.value, "function") == 0) {
            current_token.type = TOKEN_FUNCTION;
        } else if (strcmp(current_token.value, "return") == 0) {
            current_token.type = TOKEN_RETURN;
        } else {
            current_token.type = TOKEN_VARIABLE;
        }
        return;
    }
    
    switch (input[pos]) {
        case '+': case '-': case '*': case '/':
            current_token.type = TOKEN_OPERATOR;
            current_token.value[0] = input[pos++];
            current_token.value[1] = '\0';
            break;
        case '<': case '>':
            current_token.type = TOKEN_COMPARISON;
            current_token.value[0] = input[pos++];
            current_token.value[1] = '\0';
            break;
        case ';':
            current_token.type = TOKEN_SEMICOLON;
            pos++;
            break;
        case '(':
            current_token.type = TOKEN_LPAREN;
            pos++;
            break;
        case ')':
            current_token.type = TOKEN_RPAREN;
            pos++;
            break;
        case '{':
            current_token.type = TOKEN_LBRACE;
            pos++;
            break;
        case '}':
            current_token.type = TOKEN_RBRACE;
            pos++;
            break;
        case '=':
            current_token.type = TOKEN_EQUALS;
            pos++;
            break;
        case ',':
            current_token.type = TOKEN_COMMA;
            pos++;
            break;
        default:
            printf("Error: Unknown character '%c'\n", input[pos]);
            exit(1);
    }
}

// Function management
int find_function(const char *name) {
    for (int i = 0; i < func_count; i++) {
        if (strcmp(functions[i].name, name) == 0) {
            return i;
        }
    }
    return -1;
}

// Parse function parameters
void parse_parameters(Function *func) {
    func->param_count = 0;
    
    if (current_token.type == TOKEN_RPAREN) {
        return;
    }
    
    do {
        if (current_token.type != TOKEN_VARIABLE) {
            printf("Error: Expected parameter name\n");
            exit(1);
        }
        
        strcpy(func->params[func->param_count].name, current_token.value);
        func->param_count++;
        get_next_token();
        
        if (current_token.type == TOKEN_RPAREN) {
            break;
        }
        
        if (current_token.type != TOKEN_COMMA) {
            printf("Error: Expected ',' between parameters\n");
            exit(1);
        }
        get_next_token();
    } while (func->param_count < MAX_PARAMS);
}

// Parse function definition
int parse_function_definition() {
    get_next_token();  // 跳過 'function' 關鍵字
    
    if (current_token.type != TOKEN_VARIABLE) {
        printf("Error: Expected function name\n");
        exit(1);
    }
    
    Function func;
    strcpy(func.name, current_token.value);
    get_next_token();
    
    if (current_token.type != TOKEN_LPAREN) {
        printf("Error: Expected '(' after function name\n");
        exit(1);
    }
    get_next_token();
    
    parse_parameters(&func);
    
    if (current_token.type != TOKEN_RPAREN) {
        printf("Error: Expected ')' after parameters\n");
        exit(1);
    }
    get_next_token();
    
    func.start_pos = pos;  // 儲存函數體開始位置
    
    // 跳過函數體
    int brace_count = 1;
    while (brace_count > 0 && input[pos] != '\0') {
        if (input[pos] == '{') brace_count++;
        if (input[pos] == '}') brace_count--;
        pos++;
    }
    
    functions[func_count++] = func;
    get_next_token();
    
    return 0;
}

// Execute function
int execute_function(int func_idx, int *args, int arg_count) {
    // 保存當前執行環境
    int old_pos = pos;
    char *old_input = input;
    int old_var_count = var_count;
    
    // 設置新的執行環境
    pos = functions[func_idx].start_pos;
    input = old_input;
    
    // 創建參數變數
    for (int i = 0; i < functions[func_idx].param_count; i++) {
        int var_idx = find_or_create_variable(functions[func_idx].params[i].name);
        variables[var_idx].value = args[i];
    }
    
    // 執行函數體
    get_next_token();  // 跳過 '{'
    int result = parse_block();
    
    // 恢復原來的執行環境
    pos = old_pos;
    var_count = old_var_count;
    
    return result;
}

// Parse function call
int parse_function_call(const char *func_name) {
    int func_idx = find_function(func_name);
    if (func_idx == -1) {
        printf("Error: Undefined function '%s'\n", func_name);
        exit(1);
    }
    
    get_next_token();  // 跳過 '('
    
    // 解析參數
    int args[MAX_PARAMS];
    int arg_count = 0;
    
    if (current_token.type != TOKEN_RPAREN) {
        do {
            args[arg_count++] = parse_expression();
            
            if (current_token.type == TOKEN_RPAREN) {
                break;
            }
            
            if (current_token.type != TOKEN_COMMA) {
                printf("Error: Expected ',' between arguments\n");
                exit(1);
            }
            get_next_token();
        } while (arg_count < MAX_PARAMS);
    }
    
    if (arg_count != functions[func_idx].param_count) {
        printf("Error: Wrong number of arguments for function '%s'\n", func_name);
        exit(1);
    }
    
    get_next_token();  // 跳過 ')'
    
    return execute_function(func_idx, args, arg_count);
}

// Parse expression (更新以支援函數呼叫)
int parse_expression() {
    int left;
    
    if (current_token.type == TOKEN_NUMBER) {
        left = atoi(current_token.value);
        get_next_token();
    } else if (current_token.type == TOKEN_VARIABLE) {
        char var_name[MAX_TOKEN_LEN];
        strcpy(var_name, current_token.value);
        get_next_token();
        
        if (current_token.type == TOKEN_LPAREN) {
            // 這是一個函數呼叫
            left = parse_function_call(var_name);
        } else {
            // 這是一個變數
            int var_idx = find_or_create_variable(var_name);
            left = variables[var_idx].value;
        }
    } else {
        printf("Error: Expected number or variable\n");
        exit(1);
    }
    
    while (current_token.type == TOKEN_OPERATOR || current_token.type == TOKEN_COMPARISON) {
        char op = current_token.value[0];
        get_next_token();
        
        int right = parse_expression();
        
        switch (op) {
            case '+': left += right; break;
            case '-': left -= right; break;
            case '*': left *= right; break;
            case '/': 
                if (right == 0) {
                    printf("Error: Division by zero\n");
                    exit(1);
                }
                left /= right; 
                break;
            case '<': left = left < right; break;
            case '>': left = left > right; break;
        }
    }
    
    return left;
}

// Parse statement (更新以支援函數定義)
int parse_statement() {
    int value = 0;
    
    if (current_token.type == TOKEN_FUNCTION) {
        return parse_function_definition();
    } else if (current_token.type == TOKEN_IF) {
        get_next_token();
        
        if (current_token.type != TOKEN_LPAREN) {
            printf("Error: Expected '('\n");
            exit(1);
        }
        get_next_token();
        
        int condition = parse_expression();
        
        if (current_token.type != TOKEN_RPAREN) {
            printf("Error: Expected ')'\n");
            exit(1);
        }
        get_next_token();
        
        if (condition) {
            value = parse_block();
        } else {
            parse_block();
        }
    } else if (current_token.type == TOKEN_WHILE) {
        get_next_token();
        
        if (current_token.type != TOKEN_LPAREN) {
            printf("Error: Expected '('\n");
            exit(1);
        }
        get_next_token();
        
        int start_pos = pos;
        Token start_token = current_token;
        
        int condition = parse_expression();
        
        if (current_token.type != TOKEN_RPAREN) {
            printf("Error: Expected ')'\n");
            exit(1);
        }
        get_next_token();
        
        while (condition) {
            value = parse_block();
            
            pos = start_pos;
            current_token = start_token;
            condition = parse_expression();
            
            if (current_token.type != TOKEN_RPAREN) {
                printf("Error: Expected ')'\n");
                exit(1);
            }
            get_next_token();
        }
        parse_block();
    } else if (current_token.type == TOKEN_RETURN) {
        get_next_token();
        value = parse_expression();
        
        if (current_token.type != TOKEN_SEMICOLON) {
            printf("Error: Expected ';'\n");
            exit(1);
        }
        get_next_token();
    } else {
        value = parse_assignment();
        
        if (current_token.type != TOKEN_SEMICOLON) {
            printf("Error: Expected ';'\n");
            exit(1);
        }
        get_next_token();
    }
    
    return value;
}

// Find or create variable
int find_or_create_variable(const char *name) {
    for (int i = 0; i < var_count; i++) {
        if (strcmp(variables[i].name, name) == 0) {
            return i;
        }
    }
    
    strcpy(variables[var_count].name, name);
    variables[var_count].value = 0;
    return var_count++;
}

// Parse assignment
int parse_assignment() {
    if (current_token.type != TOKEN_VARIABLE) {
        printf("Error: Expected variable name\n");
        exit(1);
    }
    
    int var_idx = find_or_create_variable(current_token.value);
    get_next_token();
    
    if (current_token.type != TOKEN_EQUALS) {
        printf("Error: Expected '='\n");
        exit(1);
    }
    get_next_token();
    
    int value = parse_expression();
    variables[var_idx].value = value;
    
    return value;
}

// Parse block
int parse_block() {
    if (current_token.type != TOKEN_LBRACE) {
        printf("Error: Expected '{'\n");
        exit(1);
    }
    get_next_token();
    
    int last_value = 0;
    while (current_token.type != TOKEN_RBRACE && current_token.type != TOKEN_EOF) {
        last_value = parse_statement();
    }
    
    if (current_token.type != TOKEN_RBRACE) {
        printf("Error: Expected '}'\n");
        exit(1);
    }
    get_next_token();
    
    return last_value;
}

// Parse program
int parse_program() {
    int last_value = 0;
    get_next_token();
    
    while (current_token.type != TOKEN_EOF) {
        last_value = parse_statement();
    }
    
    return last_value;
}

// Main function with example
int main() {
    // Example program with function
    char program[] = 
        "function add(a, b) {\n"
        "    return a + b;\n"
        "}\n"
        "x = 5;\n"
        "y = 10;\n"
        "z = add(x, y);\n"
        "if (z > 10) {\n"
        "    w = z * 2;\n"
        "}\n";
    
    input = program;
    int result = parse_program();
    
    // Print all variables
    printf("Final variable values:\n");
    for (int i = 0; i < var_count; i++) {
        printf("%s = %d\n", variables[i].name, variables[i].value);
    }
    
    return 0;
}