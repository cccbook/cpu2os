#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

//======================================================================
// 1. 定義 (Token, AST, IR)
//======================================================================

// --- Token 類型 ---
typedef enum {
    TOKEN_EOF, TOKEN_ILLEGAL, TOKEN_INTEGER, TOKEN_IDENTIFIER,
    TOKEN_ASSIGN, TOKEN_PLUS, TOKEN_MINUS, TOKEN_MUL, TOKEN_DIV,
    TOKEN_EQ, TOKEN_NE, TOKEN_LT, TOKEN_GT, TOKEN_SEMI,
    TOKEN_LPAREN, TOKEN_RPAREN, TOKEN_LBRACE, TOKEN_RBRACE,
    TOKEN_LET, TOKEN_IF, TOKEN_ELSE, TOKEN_WHILE,
} TokenType;

// --- Token 結構 ---
typedef struct {
    TokenType type;
    char literal[100];
    int line;
    int col;
} Token;

// --- AST 節點類型與結構 ---
typedef enum {
    NODE_TYPE_PROGRAM, NODE_TYPE_NUM, NODE_TYPE_IDENTIFIER, NODE_TYPE_BIN_OP,
    NODE_TYPE_ASSIGN, NODE_TYPE_VAR_DECL, NODE_TYPE_IF, NODE_TYPE_WHILE,
    NODE_TYPE_BLOCK,
} NodeType;

typedef struct ASTNode {
    NodeType type;
    struct ASTNode *left;
    struct ASTNode *right;
    Token token;
    struct ASTNode *condition;
    struct ASTNode *consequence;
    struct ASTNode *alternative;
    struct ASTNode **statements;
    int statement_count;
} ASTNode;

// --- IR 指令 ---
typedef enum {
    OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_EQ, OP_NE, OP_LT, OP_GT,
    OP_ASSIGN_REG, OP_LOAD_CONST, OP_LOAD_VAR, OP_STORE_VAR,
    OP_GOTO, OP_IF_FALSE_GOTO, OP_LABEL,
} OpCode;

typedef struct {
    OpCode opcode;
    char result[20];
    char arg1[20];
    char arg2[20];
} IR_Instruction;

//======================================================================
// 2. 全域變數與輔助函式
//======================================================================

static IR_Instruction ir_code[1024];
static int ir_count = 0;
static int temp_count = 0;
static int label_count = 0;

typedef struct {
    const char* input;
    size_t position;
    size_t read_position;
    char ch;
    int line;
    size_t line_start_pos;
} Lexer;

typedef struct {
    Lexer* lexer;
    Token current_token;
    Token peek_token;
} Parser;

void error(int line, int col, const char* message) {
    fprintf(stderr, "編譯錯誤 (行 %d, 欄 %d): %s\n", line, col, message);
    exit(1);
}

char* new_temp() {
    static char temp_name[20];
    snprintf(temp_name, sizeof(temp_name), "t%d", temp_count++);
    return strdup(temp_name);
}

char* new_label() {
    static char label_name[20];
    snprintf(label_name, sizeof(label_name), "L%d", label_count++);
    return strdup(label_name);
}

void emit(OpCode op, const char* result, const char* arg1, const char* arg2) {
    if (ir_count >= 1024) error(0, 0, "中間碼緩衝區溢位");
    ir_code[ir_count].opcode = op;
    strcpy(ir_code[ir_count].result, result ? result : "");
    strcpy(ir_code[ir_count].arg1, arg1 ? arg1 : "");
    strcpy(ir_code[ir_count].arg2, arg2 ? arg2 : "");
    ir_count++;
}


//======================================================================
// 3. 詞法分析器 (Lexer)
//======================================================================

void read_char(Lexer* l) {
    if (l->ch == '\n') {
        l->line++;
        l->line_start_pos = l->position + 1;
    }
    if (l->read_position >= strlen(l->input)) {
        l->ch = '\0';
    } else {
        l->ch = l->input[l->read_position];
    }
    l->position = l->read_position;
    l->read_position++;
}

Lexer* create_lexer(const char* input) {
    Lexer* l = (Lexer*)malloc(sizeof(Lexer));
    l->input = input;
    l->position = 0;
    l->read_position = 0;
    l->line = 1;
    l->line_start_pos = 0;
    read_char(l);
    return l;
}

void skip_whitespace(Lexer* l) {
    while (isspace(l->ch)) {
        read_char(l);
    }
}

char peek_char(Lexer* l) {
    if (l->read_position >= strlen(l->input)) return '\0';
    return l->input[l->read_position];
}

Token new_token(TokenType type, const char* literal, int line, int col) {
    Token tok;
    tok.type = type;
    strncpy(tok.literal, literal, sizeof(tok.literal) - 1);
    tok.literal[sizeof(tok.literal) - 1] = '\0';
    tok.line = line;
    tok.col = col;
    return tok;
}

Token next_token(Lexer* l) {
    skip_whitespace(l);
    int current_col = l->position - l->line_start_pos + 1;
    int current_line = l->line;
    Token tok;
    char literal[2] = {l->ch, '\0'};

    switch (l->ch) {
        case '=':
            if (peek_char(l) == '=') { read_char(l); tok = new_token(TOKEN_EQ, "==", current_line, current_col); }
            else { tok = new_token(TOKEN_ASSIGN, "=", current_line, current_col); }
            break;
        case '!':
            if (peek_char(l) == '=') { read_char(l); tok = new_token(TOKEN_NE, "!=", current_line, current_col); }
            else { tok = new_token(TOKEN_ILLEGAL, "!", current_line, current_col); }
            break;
        case '+': tok = new_token(TOKEN_PLUS, "+", current_line, current_col); break;
        case '-': tok = new_token(TOKEN_MINUS, "-", current_line, current_col); break;
        case '*': tok = new_token(TOKEN_MUL, "*", current_line, current_col); break;
        case '/': tok = new_token(TOKEN_DIV, "/", current_line, current_col); break;
        case '<': tok = new_token(TOKEN_LT, "<", current_line, current_col); break;
        case '>': tok = new_token(TOKEN_GT, ">", current_line, current_col); break;
        case ';': tok = new_token(TOKEN_SEMI, ";", current_line, current_col); break;
        case '(': tok = new_token(TOKEN_LPAREN, "(", current_line, current_col); break;
        case ')': tok = new_token(TOKEN_RPAREN, ")", current_line, current_col); break;
        case '{': tok = new_token(TOKEN_LBRACE, "{", current_line, current_col); break;
        case '}': tok = new_token(TOKEN_RBRACE, "}", current_line, current_col); break;
        case '\0': tok = new_token(TOKEN_EOF, "", current_line, current_col); break;
        default:
            if (isalpha(l->ch) || l->ch == '_') {
                size_t start_pos = l->position;
                while (isalnum(l->ch) || l->ch == '_') read_char(l);
                size_t len = l->position - start_pos;
                char ident[100];
                strncpy(ident, &l->input[start_pos], len);
                ident[len] = '\0';

                if (strcmp(ident, "let") == 0) return new_token(TOKEN_LET, "let", current_line, current_col);
                if (strcmp(ident, "if") == 0) return new_token(TOKEN_IF, "if", current_line, current_col);
                if (strcmp(ident, "else") == 0) return new_token(TOKEN_ELSE, "else", current_line, current_col);
                if (strcmp(ident, "while") == 0) return new_token(TOKEN_WHILE, "while", current_line, current_col);
                return new_token(TOKEN_IDENTIFIER, ident, current_line, current_col);
            } else if (isdigit(l->ch)) {
                size_t start_pos = l->position;
                while (isdigit(l->ch)) read_char(l);
                size_t len = l->position - start_pos;
                char num[100];
                strncpy(num, &l->input[start_pos], len);
                num[len] = '\0';
                return new_token(TOKEN_INTEGER, num, current_line, current_col);
            } else {
                tok = new_token(TOKEN_ILLEGAL, literal, current_line, current_col);
            }
    }
    if (tok.type != TOKEN_EOF) read_char(l);
    return tok;
}


//======================================================================
// 4. 語法分析器 (Parser)
//======================================================================

ASTNode* create_ast_node(NodeType type) {
    ASTNode* node = (ASTNode*)calloc(1, sizeof(ASTNode));
    node->type = type;
    return node;
}

ASTNode* parse_statement(Parser* p);
ASTNode* parse_expression(Parser* p, int precedence);

void parser_next_token(Parser* p) {
    p->current_token = p->peek_token;
    p->peek_token = next_token(p->lexer);
}

Parser* create_parser(Lexer* l) {
    Parser* p = (Parser*)malloc(sizeof(Parser));
    p->lexer = l;
    parser_next_token(p);
    parser_next_token(p);
    return p;
}

ASTNode* parse_block_statement(Parser* p) {
    ASTNode* block = create_ast_node(NODE_TYPE_BLOCK);
    block->statements = (ASTNode**)malloc(sizeof(ASTNode*) * 100);
    block->statement_count = 0;
    parser_next_token(p); // eat {
    while (p->current_token.type != TOKEN_RBRACE && p->current_token.type != TOKEN_EOF) {
        block->statements[block->statement_count++] = parse_statement(p);
        parser_next_token(p);
    }
    if (p->current_token.type != TOKEN_RBRACE) {
         error(p->current_token.line, p->current_token.col, "區塊語句缺少 '}'");
    }
    return block;
}

ASTNode* parse_let_statement(Parser* p) {
    ASTNode* node = create_ast_node(NODE_TYPE_VAR_DECL);
    parser_next_token(p); // eat let
    if (p->current_token.type != TOKEN_IDENTIFIER) error(p->current_token.line, p->current_token.col, "預期為識別碼");
    node->left = create_ast_node(NODE_TYPE_IDENTIFIER);
    node->left->token = p->current_token;
    parser_next_token(p);
    if (p->current_token.type != TOKEN_ASSIGN) error(p->current_token.line, p->current_token.col, "預期為 '='");
    parser_next_token(p);
    node->right = parse_expression(p, 0);
    if (p->peek_token.type == TOKEN_SEMI) parser_next_token(p);
    return node;
}

ASTNode* parse_if_statement(Parser *p) {
    ASTNode* node = create_ast_node(NODE_TYPE_IF);
    parser_next_token(p); // eat if
    if (p->current_token.type != TOKEN_LPAREN) error(p->current_token.line, p->current_token.col, "if 條件式缺少 '('");
    parser_next_token(p);
    node->condition = parse_expression(p, 0);

    // **** BUG FIX IS HERE ****
    // We must check the *peek* token, not the current one.
    if (p->peek_token.type != TOKEN_RPAREN) error(p->peek_token.line, p->peek_token.col, "if 條件式缺少 ')'");
    parser_next_token(p); // Consume expression's last token
    parser_next_token(p); // Consume ')'

    if (p->current_token.type != TOKEN_LBRACE) error(p->current_token.line, p->current_token.col, "if 主體缺少 '{'");
    node->consequence = parse_block_statement(p);
    if (p->peek_token.type == TOKEN_ELSE) {
        parser_next_token(p); // eat '}'
        parser_next_token(p); // eat 'else'
        if (p->current_token.type != TOKEN_LBRACE) error(p->current_token.line, p->current_token.col, "else 主體缺少 '{'");
        node->alternative = parse_block_statement(p);
    }
    return node;
}

ASTNode* parse_while_statement(Parser *p) {
    ASTNode* node = create_ast_node(NODE_TYPE_WHILE);
    parser_next_token(p); // eat while
    if (p->current_token.type != TOKEN_LPAREN) error(p->current_token.line, p->current_token.col, "while 條件式缺少 '('");
    parser_next_token(p);
    node->condition = parse_expression(p, 0);

    // **** BUG FIX IS HERE ****
    // We must check the *peek* token, not the current one.
    if (p->peek_token.type != TOKEN_RPAREN) error(p->peek_token.line, p->peek_token.col, "while 條件式缺少 ')'");
    parser_next_token(p); // Consume expression's last token
    parser_next_token(p); // Consume ')'

    if (p->current_token.type != TOKEN_LBRACE) error(p->current_token.line, p->current_token.col, "while 主體缺少 '{'");
    node->consequence = parse_block_statement(p);
    return node;
}

int get_precedence(TokenType type) {
    switch(type) {
        case TOKEN_EQ: case TOKEN_NE: return 1;
        case TOKEN_LT: case TOKEN_GT: return 2;
        case TOKEN_PLUS: case TOKEN_MINUS: return 3;
        case TOKEN_MUL: case TOKEN_DIV: return 4;
        default: return 0;
    }
}

ASTNode* parse_expression(Parser* p, int precedence) {
    ASTNode* left = NULL;
    if (p->current_token.type == TOKEN_INTEGER) {
        left = create_ast_node(NODE_TYPE_NUM);
        left->token = p->current_token;
    } else if (p->current_token.type == TOKEN_IDENTIFIER) {
        left = create_ast_node(NODE_TYPE_IDENTIFIER);
        left->token = p->current_token;
    } else if (p->current_token.type == TOKEN_LPAREN) {
        parser_next_token(p);
        left = parse_expression(p, 0);
        if (p->peek_token.type != TOKEN_RPAREN) error(p->peek_token.line, p->peek_token.col, "表達式缺少 ')'");
        parser_next_token(p);
    } else {
        error(p->current_token.line, p->current_token.col, "無效的表達式起始符號");
    }
    while (p->peek_token.type != TOKEN_SEMI && precedence < get_precedence(p->peek_token.type)) {
        parser_next_token(p);
        ASTNode* infix = create_ast_node(NODE_TYPE_BIN_OP);
        infix->token = p->current_token;
        infix->left = left;
        parser_next_token(p);
        infix->right = parse_expression(p, get_precedence(infix->token.type));
        left = infix;
    }
    return left;
}

ASTNode* parse_assignment_statement(Parser* p) {
    ASTNode* node = create_ast_node(NODE_TYPE_ASSIGN);
    node->left = create_ast_node(NODE_TYPE_IDENTIFIER);
    node->left->token = p->current_token;
    parser_next_token(p);
    if (p->current_token.type != TOKEN_ASSIGN) error(p->current_token.line, p->current_token.col, "預期為 '='");
    parser_next_token(p);
    node->right = parse_expression(p, 0);
    if (p->peek_token.type == TOKEN_SEMI) parser_next_token(p);
    return node;
}

ASTNode* parse_statement(Parser* p) {
    switch (p->current_token.type) {
        case TOKEN_LET: return parse_let_statement(p);
        case TOKEN_IF: return parse_if_statement(p);
        case TOKEN_WHILE: return parse_while_statement(p);
        case TOKEN_LBRACE: return parse_block_statement(p);
        case TOKEN_IDENTIFIER:
            if (p->peek_token.type == TOKEN_ASSIGN) return parse_assignment_statement(p);
        default:
            error(p->current_token.line, p->current_token.col, "無效的語句起始符號");
            return NULL;
    }
}

ASTNode* parse_program(Parser* p) {
    ASTNode* program = create_ast_node(NODE_TYPE_PROGRAM);
    program->statements = (ASTNode**)malloc(sizeof(ASTNode*) * 100);
    program->statement_count = 0;
    while (p->current_token.type != TOKEN_EOF) {
        program->statements[program->statement_count++] = parse_statement(p);
        parser_next_token(p);
    }
    return program;
}

//======================================================================
// 5. 中間碼生成器 (IR Generator) & 6. 主程式
//======================================================================

static char symbol_table[100][100];
static int symbol_count = 0;

void add_symbol(const char* name) {
    for (int i=0; i<symbol_count; ++i) if (strcmp(symbol_table[i], name)==0) return;
    strncpy(symbol_table[symbol_count], name, sizeof(symbol_table[0]) - 1);
    symbol_table[symbol_count][sizeof(symbol_table[0]) - 1] = '\0';
    symbol_count++;
}

int find_symbol(const char* name) {
    for (int i = 0; i < symbol_count; i++) if (strcmp(symbol_table[i], name) == 0) return i;
    return -1;
}

char* generate_ir(ASTNode* node) {
    if (!node) return NULL;
    char* left_reg = NULL, *right_reg = NULL, *cond_reg = NULL;

    switch (node->type) {
        case NODE_TYPE_PROGRAM: case NODE_TYPE_BLOCK:
            for (int i = 0; i < node->statement_count; ++i) generate_ir(node->statements[i]);
            break;
        case NODE_TYPE_NUM: {
            char* temp = new_temp();
            emit(OP_LOAD_CONST, temp, node->token.literal, NULL);
            return temp;
        }
        case NODE_TYPE_IDENTIFIER: {
            if (find_symbol(node->token.literal) == -1) error(node->token.line, node->token.col, "未宣告的變數");
            char* temp = new_temp();
            emit(OP_LOAD_VAR, temp, node->token.literal, NULL);
            return temp;
        }
        case NODE_TYPE_BIN_OP: {
            left_reg = generate_ir(node->left);
            right_reg = generate_ir(node->right);
            char* result_reg = new_temp();
            OpCode op;
            switch (node->token.type) {
                case TOKEN_PLUS: op = OP_ADD; break; case TOKEN_MINUS: op = OP_SUB; break;
                case TOKEN_MUL: op = OP_MUL; break; case TOKEN_DIV: op = OP_DIV; break;
                case TOKEN_EQ: op = OP_EQ; break; case TOKEN_NE: op = OP_NE; break;
                case TOKEN_LT: op = OP_LT; break; case TOKEN_GT: op = OP_GT; break;
                default:
                    error(node->token.line, node->token.col, "未知的二元運算子");
                    return NULL;
            }
            emit(op, result_reg, left_reg, right_reg);
            free(left_reg); free(right_reg);
            return result_reg;
        }
        case NODE_TYPE_VAR_DECL:
            add_symbol(node->left->token.literal);
            right_reg = generate_ir(node->right);
            emit(OP_STORE_VAR, node->left->token.literal, right_reg, NULL);
            free(right_reg);
            break;
        case NODE_TYPE_ASSIGN:
            if (find_symbol(node->left->token.literal) == -1) error(node->left->token.line, node->left->token.col, "對未宣告的變數賦值");
            right_reg = generate_ir(node->right);
            emit(OP_STORE_VAR, node->left->token.literal, right_reg, NULL);
            free(right_reg);
            break;
        case NODE_TYPE_IF: {
            cond_reg = generate_ir(node->condition);
            char* else_label = new_label();
            char* end_label = new_label();
            emit(OP_IF_FALSE_GOTO, NULL, cond_reg, else_label);
            free(cond_reg);
            generate_ir(node->consequence);
            emit(OP_GOTO, end_label, NULL, NULL);
            emit(OP_LABEL, else_label, NULL, NULL);
            if (node->alternative) generate_ir(node->alternative);
            emit(OP_LABEL, end_label, NULL, NULL);
            free(else_label); free(end_label);
            break;
        }
        case NODE_TYPE_WHILE: {
            char* start_label = new_label();
            char* end_label = new_label();
            emit(OP_LABEL, start_label, NULL, NULL);
            cond_reg = generate_ir(node->condition);
            emit(OP_IF_FALSE_GOTO, NULL, cond_reg, end_label);
            free(cond_reg);
            generate_ir(node->consequence);
            emit(OP_GOTO, start_label, NULL, NULL);
            emit(OP_LABEL, end_label, NULL, NULL);
            free(start_label); free(end_label);
            break;
        }
        default: break;
    }
    return NULL;
}

void print_ir() {
    for (int i = 0; i < ir_count; i++) {
        printf("%3d: ", i);
        switch (ir_code[i].opcode) {
            case OP_ADD: printf("%s = %s + %s\n", ir_code[i].result, ir_code[i].arg1, ir_code[i].arg2); break;
            case OP_SUB: printf("%s = %s - %s\n", ir_code[i].result, ir_code[i].arg1, ir_code[i].arg2); break;
            case OP_MUL: printf("%s = %s * %s\n", ir_code[i].result, ir_code[i].arg1, ir_code[i].arg2); break;
            case OP_DIV: printf("%s = %s / %s\n", ir_code[i].result, ir_code[i].arg1, ir_code[i].arg2); break;
            case OP_EQ: printf("%s = %s == %s\n", ir_code[i].result, ir_code[i].arg1, ir_code[i].arg2); break;
            case OP_NE: printf("%s = %s != %s\n", ir_code[i].result, ir_code[i].arg1, ir_code[i].arg2); break;
            case OP_LT: printf("%s = %s < %s\n", ir_code[i].result, ir_code[i].arg1, ir_code[i].arg2); break;
            case OP_GT: printf("%s = %s > %s\n", ir_code[i].result, ir_code[i].arg1, ir_code[i].arg2); break;
            case OP_LOAD_CONST: printf("%s = %s\n", ir_code[i].result, ir_code[i].arg1); break;
            case OP_LOAD_VAR: printf("%s = %s\n", ir_code[i].result, ir_code[i].arg1); break;
            case OP_STORE_VAR: printf("%s = %s\n", ir_code[i].result, ir_code[i].arg1); break;
            case OP_GOTO: printf("goto %s\n", ir_code[i].result); break;
            case OP_IF_FALSE_GOTO: printf("if_false %s goto %s\n", ir_code[i].arg1, ir_code[i].arg2); break;
            case OP_LABEL: printf("%s:\n", ir_code[i].result); break;
            default: break;
        }
    }
}

int main() {
    const char* source_code =
        "let result = 0;\n"
        "let i = 10;\n"
        "\n"
        "while (i > 0) {\n"
        "  result = result + i;\n"
        "  i = i - 1;\n"
        "}\n"
        "\n"
        "let final_val = 0;\n"
        "if (result > 50) {\n"
        "   final_val = 1;\n"
        "} else {\n"
        "   final_val = 2;\n"
        "}\n";

    printf("原始碼:\n---\n%s---\n\n", source_code);

    Lexer* lexer = create_lexer(source_code);
    Parser* parser = create_parser(lexer);
    ASTNode* ast = parse_program(parser);

    printf("生成的中間碼:\n---\n");
    generate_ir(ast);
    print_ir();
    
    // 省略記憶體釋放程式碼...

    return 0;
}
