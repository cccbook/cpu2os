#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_TOKEN_LEN 256
#define MAX_TOKENS 1000
#define MAX_TEMP_VARS 100
#define MAX_CODE_LINES 1000
#define MAX_LINE_LEN 256

// Token 類型
typedef enum {
    TOKEN_NUMBER,
    TOKEN_PLUS,
    TOKEN_MINUS,
    TOKEN_MULTIPLY,
    TOKEN_DIVIDE,
    TOKEN_LPAREN,
    TOKEN_RPAREN,
    TOKEN_PRINT,
    TOKEN_STRING,
    TOKEN_COMMA,
    TOKEN_EOF
} TokenType;

// Token 結構
typedef struct {
    TokenType type;
    char value[MAX_TOKEN_LEN];
} Token;

// 詞法分析器狀態
typedef struct {
    const char* input;
    int pos;
    Token tokens[MAX_TOKENS];
    int token_count;
} Lexer;

// 抽象語法樹節點類型
typedef enum {
    NODE_NUMBER,
    NODE_BINARY_OP,
    NODE_STRING,
    NODE_PRINT
} NodeType;

// 抽象語法樹節點
typedef struct Node {
    NodeType type;
    char value[MAX_TOKEN_LEN];
    struct Node* left;
    struct Node* right;
    struct Node** print_args;  // for print statement
    int arg_count;            // for print statement
} Node;

// 中間碼指令類型
typedef enum {
    INSTR_ASSIGN,
    INSTR_PRINT
} InstructionType;

// 中間碼指令
typedef struct {
    InstructionType type;
    char result[MAX_TOKEN_LEN];
    char op1[MAX_TOKEN_LEN];
    char operator;
    char op2[MAX_TOKEN_LEN];
    char** print_args;
    int arg_count;
} Instruction;

// 編譯器狀態
typedef struct {
    Instruction instructions[MAX_CODE_LINES];
    int instruction_count;
    int temp_counter;
} Compiler;

// 解譯器變數表
typedef struct {
    char name[MAX_TOKEN_LEN];
    double value;
} Variable;

// 解譯器狀態
typedef struct {
    Variable variables[MAX_TEMP_VARS];
    int var_count;
} Interpreter;

// 詞法分析器函數
void init_lexer(Lexer* lexer, const char* input) {
    lexer->input = input;
    lexer->pos = 0;
    lexer->token_count = 0;
}

int is_whitespace(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

void skip_whitespace(Lexer* lexer) {
    while (is_whitespace(lexer->input[lexer->pos])) {
        lexer->pos++;
    }
}

void add_token(Lexer* lexer, TokenType type, const char* value) {
    Token token;
    token.type = type;
    strncpy(token.value, value, MAX_TOKEN_LEN - 1);
    token.value[MAX_TOKEN_LEN - 1] = '\0';
    lexer->tokens[lexer->token_count++] = token;
}

void tokenize(Lexer* lexer) {
    while (lexer->input[lexer->pos] != '\0') {
        skip_whitespace(lexer);
        
        if (lexer->input[lexer->pos] == '\0') break;
        
        // 數字
        if (isdigit(lexer->input[lexer->pos])) {
            char number[MAX_TOKEN_LEN];
            int i = 0;
            while (isdigit(lexer->input[lexer->pos]) || lexer->input[lexer->pos] == '.') {
                number[i++] = lexer->input[lexer->pos++];
            }
            number[i] = '\0';
            add_token(lexer, TOKEN_NUMBER, number);
            continue;
        }
        
        // 字串
        if (lexer->input[lexer->pos] == '"') {
            char string[MAX_TOKEN_LEN];
            int i = 0;
            lexer->pos++; // 跳過開頭的引號
            while (lexer->input[lexer->pos] != '"' && lexer->input[lexer->pos] != '\0') {
                string[i++] = lexer->input[lexer->pos++];
            }
            string[i] = '\0';
            lexer->pos++; // 跳過結尾的引號
            add_token(lexer, TOKEN_STRING, string);
            continue;
        }
        
        // 運算符和其他符號
        switch (lexer->input[lexer->pos]) {
            case '+': add_token(lexer, TOKEN_PLUS, "+"); break;
            case '-': add_token(lexer, TOKEN_MINUS, "-"); break;
            case '*': add_token(lexer, TOKEN_MULTIPLY, "*"); break;
            case '/': add_token(lexer, TOKEN_DIVIDE, "/"); break;
            case '(': add_token(lexer, TOKEN_LPAREN, "("); break;
            case ')': add_token(lexer, TOKEN_RPAREN, ")"); break;
            case ',': add_token(lexer, TOKEN_COMMA, ","); break;
            default:
                if (strncmp(lexer->input + lexer->pos, "print", 5) == 0) {
                    add_token(lexer, TOKEN_PRINT, "print");
                    lexer->pos += 4;
                }
                break;
        }
        lexer->pos++;
    }
    add_token(lexer, TOKEN_EOF, "");
}

// 語法分析器函數
Node* create_node(NodeType type, const char* value) {
    Node* node = (Node*)malloc(sizeof(Node));
    node->type = type;
    strncpy(node->value, value, MAX_TOKEN_LEN - 1);
    node->value[MAX_TOKEN_LEN - 1] = '\0';
    node->left = NULL;
    node->right = NULL;
    node->print_args = NULL;
    node->arg_count = 0;
    return node;
}

void free_node(Node* node) {
    if (node == NULL) return;
    free_node(node->left);
    free_node(node->right);
    if (node->print_args != NULL) {
        for (int i = 0; i < node->arg_count; i++) {
            free_node(node->print_args[i]);
        }
        free(node->print_args);
    }
    free(node);
}

// 編譯器函數
void init_compiler(Compiler* compiler) {
    compiler->instruction_count = 0;
    compiler->temp_counter = 0;
}

void add_instruction(Compiler* compiler, Instruction instr) {
    compiler->instructions[compiler->instruction_count++] = instr;
}

void compile_node(Compiler* compiler, Node* node, char* result) {
    if (node == NULL) return;
    
    switch (node->type) {
        case NODE_NUMBER:
        case NODE_STRING:
            strcpy(result, node->value);
            break;
            
        case NODE_BINARY_OP: {
            char left_result[MAX_TOKEN_LEN];
            char right_result[MAX_TOKEN_LEN];
            
            compile_node(compiler, node->left, left_result);
            compile_node(compiler, node->right, right_result);
            
            sprintf(result, "t%d", compiler->temp_counter++);
            
            Instruction instr;
            instr.type = INSTR_ASSIGN;
            strcpy(instr.result, result);
            strcpy(instr.op1, left_result);
            instr.operator = node->value[0];
            strcpy(instr.op2, right_result);
            
            add_instruction(compiler, instr);
            break;
        }
        
        case NODE_PRINT: {
            Instruction instr;
            instr.type = INSTR_PRINT;
            instr.arg_count = node->arg_count;
            instr.print_args = (char**)malloc(sizeof(char*) * node->arg_count);
            
            for (int i = 0; i < node->arg_count; i++) {
                instr.print_args[i] = (char*)malloc(MAX_TOKEN_LEN);
                char temp_result[MAX_TOKEN_LEN];
                compile_node(compiler, node->print_args[i], temp_result);
                strcpy(instr.print_args[i], temp_result);
            }
            
            add_instruction(compiler, instr);
            break;
        }
    }
}

// 解譯器函數
void init_interpreter(Interpreter* interpreter) {
    interpreter->var_count = 0;
}

double get_variable_value(Interpreter* interpreter, const char* name) {
    // 如果是數字，直接返回
    char* endptr;
    double value = strtod(name, &endptr);
    if (*endptr == '\0') return value;
    
    // 在變數表中查找
    for (int i = 0; i < interpreter->var_count; i++) {
        if (strcmp(interpreter->variables[i].name, name) == 0) {
            return interpreter->variables[i].value;
        }
    }
    printf("Error: Variable %s not found\n", name);
    return 0;
}

void set_variable_value(Interpreter* interpreter, const char* name, double value) {
    // 查找是否已存在
    for (int i = 0; i < interpreter->var_count; i++) {
        if (strcmp(interpreter->variables[i].name, name) == 0) {
            interpreter->variables[i].value = value;
            return;
        }
    }
    
    // 不存在則新增
    if (interpreter->var_count < MAX_TEMP_VARS) {
        strcpy(interpreter->variables[interpreter->var_count].name, name);
        interpreter->variables[interpreter->var_count].value = value;
        interpreter->var_count++;
    }
}

void execute_instruction(Interpreter* interpreter, Instruction* instr) {
    switch (instr->type) {
        case INSTR_ASSIGN: {
            double op1 = get_variable_value(interpreter, instr->op1);
            double op2 = get_variable_value(interpreter, instr->op2);
            double result;
            
            switch (instr->operator) {
                case '+': result = op1 + op2; break;
                case '-': result = op1 - op2; break;
                case '*': result = op1 * op2; break;
                case '/': result = op1 / op2; break;
                default:
                    printf("Error: Unknown operator %c\n", instr->operator);
                    return;
            }
            
            set_variable_value(interpreter, instr->result, result);
            break;
        }
        
        case INSTR_PRINT: {
            for (int i = 0; i < instr->arg_count; i++) {
                if (i > 0) printf(" ");
                
                // 判斷是否為字串（以引號開頭）
                if (instr->print_args[i][0] == '"') {
                    // 移除引號並打印字串
                    printf("%s", instr->print_args[i] + 1);
                } else {
                    // 打印數值
                    double value = get_variable_value(interpreter, instr->print_args[i]);
                    printf("%g", value);
                }
            }
            printf("\n");
            break;
        }
    }
}

// 主程序
int main() {
    const char* source_code = "print(\"3+(2*5)=\", 3+(2*5))";
    
    // 初始化詞法分析器
    Lexer lexer;
    init_lexer(&lexer, source_code);
    
    // 進行詞法分析
    tokenize(&lexer);
    
    // 初始化編譯器
    Compiler compiler;
    init_compiler(&compiler);
    
    // 生成中間碼
    Node* ast = create_node(NODE_PRINT, "print");
    // ... 這裡應該有語法分析並建立AST的代碼 ...
    
    char result[MAX_TOKEN_LEN];
    compile_node(&compiler, ast, result);
    
    // 初始化解譯器
    Interpreter interpreter;
    init_interpreter(&interpreter);
    
    // 執行中間碼
    for (int i = 0; i < compiler.instruction_count; i++) {
        execute_instruction(&interpreter, &compiler.instructions[i]);
    }
    
    // 釋放記憶體
    free_node(ast);
    
    return 0;
}