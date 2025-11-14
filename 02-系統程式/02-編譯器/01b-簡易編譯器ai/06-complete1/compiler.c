// ==========================================================
// file: compiler.c (Expanded with if/while support)
// ==========================================================
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "ir.h"

// --- Token 類型 ---
typedef enum
{
    TOKEN_EOF,
    TOKEN_ILLEGAL,
    TOKEN_INTEGER,
    TOKEN_IDENTIFIER,
    TOKEN_ASSIGN,
    TOKEN_PLUS,
    TOKEN_MINUS,
    TOKEN_MUL,
    TOKEN_DIV,
    TOKEN_EQ,
    TOKEN_NE,
    TOKEN_LT,
    TOKEN_GT,
    TOKEN_SEMI,
    TOKEN_LPAREN,
    TOKEN_RPAREN,
    TOKEN_LBRACE,
    TOKEN_RBRACE,
    TOKEN_COMMA,
    TOKEN_LET,
    TOKEN_IF,
    TOKEN_ELSE,
    TOKEN_WHILE,
    TOKEN_FN,
    TOKEN_RETURN,
} TokenType;

// --- Token 結構 ---
typedef struct
{
    TokenType type;
    char literal[100];
    int line;
    int col;
} Token;

// --- AST 節點類型 ---
typedef enum
{
    NODE_TYPE_PROGRAM,
    NODE_TYPE_NUM,
    NODE_TYPE_IDENTIFIER,
    NODE_TYPE_BIN_OP,
    NODE_TYPE_ASSIGN,
    NODE_TYPE_VAR_DECL,
    NODE_TYPE_IF,
    NODE_TYPE_WHILE,
    NODE_TYPE_BLOCK,
    NODE_TYPE_FUNC_DECL,
    NODE_TYPE_CALL,
    NODE_TYPE_RETURN,
} NodeType;

// --- AST 節點結構 ---
typedef struct ASTNode
{
    NodeType type;
    Token token;
    struct ASTNode *left;
    struct ASTNode *right;
    struct ASTNode **params;
    int param_count;
    struct ASTNode *body;
    struct ASTNode **arguments;
    int argument_count;
    struct ASTNode *condition;
    struct ASTNode *consequence;
    struct ASTNode *alternative;
    struct ASTNode **statements;
    int statement_count;
} ASTNode;

//======================================================================
// 2. 全域變數與輔助函式
//======================================================================

static IR_Instruction ir_code[1024];
static int ir_count = 0;
static int temp_count = 0;
static int label_count = 0;
typedef struct
{
    const char *input;
    size_t position, read_position;
    char ch;
    int line;
    size_t line_start_pos;
} Lexer;
typedef struct
{
    Lexer *lexer;
    Token current_token, peek_token;
} Parser;

void error(int line, int col, const char *message)
{
    fprintf(stderr, "編譯錯誤 (行 %d, 欄 %d): %s\n", line, col, message);
    exit(1);
}
char *new_temp()
{
    static char name[20];
    snprintf(name, sizeof(name), "t%d", temp_count++);
    return strdup(name);
}
char *new_label()
{
    static char name[20];
    snprintf(name, sizeof(name), "L%d", label_count++);
    return strdup(name);
}
void emit(OpCode op, const char *result, const char *arg1, const char *arg2)
{
    if (ir_count >= 1024)
        error(0, 0, "中間碼緩衝區溢位");
    ir_code[ir_count].opcode = op;
    strncpy(ir_code[ir_count].result, result ? result : "", sizeof(ir_code[0].result) - 1);
    strncpy(ir_code[ir_count].arg1, arg1 ? arg1 : "", sizeof(ir_code[0].arg1) - 1);
    strncpy(ir_code[ir_count].arg2, arg2 ? arg2 : "", sizeof(ir_code[0].arg2) - 1);
    ir_count++;
}

//======================================================================
// 3. 詞法分析器 (Lexer)
//======================================================================
void read_char(Lexer *l)
{
    if (l->ch == '\n')
    {
        l->line++;
        l->line_start_pos = l->position + 1;
    }
    if (l->read_position >= strlen(l->input))
    {
        l->ch = '\0';
    }
    else
    {
        l->ch = l->input[l->read_position];
    }
    l->position = l->read_position;
    l->read_position++;
}
Lexer *create_lexer(const char *input)
{
    Lexer *l = (Lexer *)malloc(sizeof(Lexer));
    l->input = input;
    l->position = l->read_position = l->line_start_pos = 0;
    l->line = 1;
    read_char(l);
    return l;
}
void skip_whitespace(Lexer *l)
{
    while (isspace(l->ch))
        read_char(l);
}
char peek_char(Lexer *l)
{
    if (l->read_position >= strlen(l->input))
        return '\0';
    return l->input[l->read_position];
}
Token new_token(TokenType type, const char *literal, int line, int col)
{
    Token tok;
    tok.type = type;
    strncpy(tok.literal, literal, sizeof(tok.literal) - 1);
    tok.literal[sizeof(tok.literal) - 1] = '\0';
    tok.line = line;
    tok.col = col;
    return tok;
}
Token next_token(Lexer *l)
{
    skip_whitespace(l);
    int current_col = l->position - l->line_start_pos + 1;
    int current_line = l->line;
    Token tok;
    char literal[2] = {l->ch, '\0'};
    switch (l->ch)
    {
    case '=':
        if (peek_char(l) == '=')
        {
            read_char(l);
            tok = new_token(TOKEN_EQ, "==", current_line, current_col);
        }
        else
        {
            tok = new_token(TOKEN_ASSIGN, "=", current_line, current_col);
        }
        break;
    case '!':
        if (peek_char(l) == '=')
        {
            read_char(l);
            tok = new_token(TOKEN_NE, "!=", current_line, current_col);
        }
        else
        {
            tok = new_token(TOKEN_ILLEGAL, "!", current_line, current_col);
        }
        break;
    case '+':
        tok = new_token(TOKEN_PLUS, "+", current_line, current_col);
        break;
    case '-':
        tok = new_token(TOKEN_MINUS, "-", current_line, current_col);
        break;
    case '*':
        tok = new_token(TOKEN_MUL, "*", current_line, current_col);
        break;
    case '/':
        tok = new_token(TOKEN_DIV, "/", current_line, current_col);
        break;
    case '<':
        tok = new_token(TOKEN_LT, "<", current_line, current_col);
        break;
    case '>':
        tok = new_token(TOKEN_GT, ">", current_line, current_col);
        break;
    case ';':
        tok = new_token(TOKEN_SEMI, ";", current_line, current_col);
        break;
    case '(':
        tok = new_token(TOKEN_LPAREN, "(", current_line, current_col);
        break;
    case ')':
        tok = new_token(TOKEN_RPAREN, ")", current_line, current_col);
        break;
    case '{':
        tok = new_token(TOKEN_LBRACE, "{", current_line, current_col);
        break;
    case '}':
        tok = new_token(TOKEN_RBRACE, "}", current_line, current_col);
        break;
    case ',':
        tok = new_token(TOKEN_COMMA, ",", current_line, current_col);
        break;
    case '\0':
        tok = new_token(TOKEN_EOF, "", current_line, current_col);
        break;
    default:
        if (isalpha(l->ch) || l->ch == '_')
        {
            size_t start_pos = l->position;
            while (isalnum(l->ch) || l->ch == '_')
                read_char(l);
            size_t len = l->position - start_pos;
            char ident[100];
            strncpy(ident, &l->input[start_pos], len);
            ident[len] = '\0';
            if (strcmp(ident, "let") == 0)
                return new_token(TOKEN_LET, "let", current_line, current_col);
            if (strcmp(ident, "if") == 0)
                return new_token(TOKEN_IF, "if", current_line, current_col);
            if (strcmp(ident, "else") == 0)
                return new_token(TOKEN_ELSE, "else", current_line, current_col);
            if (strcmp(ident, "while") == 0)
                return new_token(TOKEN_WHILE, "while", current_line, current_col);
            if (strcmp(ident, "fn") == 0)
                return new_token(TOKEN_FN, "fn", current_line, current_col);
            if (strcmp(ident, "return") == 0)
                return new_token(TOKEN_RETURN, "return", current_line, current_col);
            return new_token(TOKEN_IDENTIFIER, ident, current_line, current_col);
        }
        else if (isdigit(l->ch))
        {
            size_t start_pos = l->position;
            while (isdigit(l->ch))
                read_char(l);
            size_t len = l->position - start_pos;
            char num[100];
            strncpy(num, &l->input[start_pos], len);
            num[len] = '\0';
            return new_token(TOKEN_INTEGER, num, current_line, current_col);
        }
        else
        {
            tok = new_token(TOKEN_ILLEGAL, literal, current_line, current_col);
        }
    }
    if (tok.type != TOKEN_EOF)
        read_char(l);
    return tok;
}

//======================================================================
// 4. 語法分析器 (Parser) - *** 新增 IF/WHILE 解析 ***
//======================================================================
ASTNode *create_ast_node(NodeType type)
{
    ASTNode *node = (ASTNode *)calloc(1, sizeof(ASTNode));
    node->type = type;
    return node;
}
ASTNode *parse_statement(Parser *p);
ASTNode *parse_expression(Parser *p, int precedence);
ASTNode *parse_block_statement(Parser *p);
void parser_next_token(Parser *p)
{
    p->current_token = p->peek_token;
    p->peek_token = next_token(p->lexer);
}
Parser *create_parser(Lexer *l)
{
    Parser *p = (Parser *)malloc(sizeof(Parser));
    p->lexer = l;
    parser_next_token(p);
    parser_next_token(p);
    return p;
}
int get_precedence(TokenType type)
{
    switch (type)
    {
    case TOKEN_EQ:
    case TOKEN_NE:
        return 1;
    case TOKEN_LT:
    case TOKEN_GT:
        return 2;
    case TOKEN_PLUS:
    case TOKEN_MINUS:
        return 3;
    case TOKEN_MUL:
    case TOKEN_DIV:
        return 4;
    case TOKEN_LPAREN:
        return 5;
    default:
        return 0;
    }
}

ASTNode *parse_call_expression(Parser *p, ASTNode *function)
{
    ASTNode *node = create_ast_node(NODE_TYPE_CALL);
    node->token = function->token;
    node->left = function;
    node->arguments = malloc(sizeof(ASTNode *) * 10);
    node->argument_count = 0;
    if (p->peek_token.type == TOKEN_RPAREN)
    {
        parser_next_token(p);
        return node;
    }
    parser_next_token(p);
    node->arguments[node->argument_count++] = parse_expression(p, 0);
    while (p->peek_token.type == TOKEN_COMMA)
    {
        parser_next_token(p);
        parser_next_token(p);
        node->arguments[node->argument_count++] = parse_expression(p, 0);
    }
    if (p->peek_token.type != TOKEN_RPAREN)
        error(p->peek_token.line, p->peek_token.col, "函數呼叫缺少 ')'");
    parser_next_token(p);
    return node;
}

ASTNode *parse_expression(Parser *p, int precedence)
{
    ASTNode *left = NULL;
    if (p->current_token.type == TOKEN_INTEGER)
    {
        left = create_ast_node(NODE_TYPE_NUM);
        left->token = p->current_token;
    }
    else if (p->current_token.type == TOKEN_IDENTIFIER)
    {
        left = create_ast_node(NODE_TYPE_IDENTIFIER);
        left->token = p->current_token;
    }
    else if (p->current_token.type == TOKEN_LPAREN)
    {
        parser_next_token(p);
        left = parse_expression(p, 0);
        if (p->peek_token.type != TOKEN_RPAREN)
            error(p->peek_token.line, p->peek_token.col, "表達式缺少 ')'");
        parser_next_token(p);
    }
    else
    {
        error(p->current_token.line, p->current_token.col, "無效的表達式起始符號");
    }
    while (p->peek_token.type != TOKEN_SEMI && p->peek_token.type != TOKEN_RPAREN && p->peek_token.type != TOKEN_RBRACE && precedence < get_precedence(p->peek_token.type))
    {
        parser_next_token(p);
        if (p->current_token.type == TOKEN_LPAREN)
        {
            left = parse_call_expression(p, left);
        }
        else
        {
            ASTNode *infix = create_ast_node(NODE_TYPE_BIN_OP);
            infix->token = p->current_token;
            infix->left = left;
            int current_precedence = get_precedence(p->current_token.type);
            parser_next_token(p);
            infix->right = parse_expression(p, current_precedence);
            left = infix;
        }
    }
    return left;
}

ASTNode *parse_return_statement(Parser *p)
{
    ASTNode *node = create_ast_node(NODE_TYPE_RETURN);
    node->token = p->current_token;
    parser_next_token(p);
    node->left = parse_expression(p, 0);
    if (p->peek_token.type == TOKEN_SEMI)
        parser_next_token(p);
    return node;
}
ASTNode *parse_let_statement(Parser *p)
{
    ASTNode *node = create_ast_node(NODE_TYPE_VAR_DECL);
    parser_next_token(p);
    if (p->current_token.type != TOKEN_IDENTIFIER)
        error(p->current_token.line, p->current_token.col, "預期為識別碼");
    node->left = create_ast_node(NODE_TYPE_IDENTIFIER);
    node->left->token = p->current_token;
    parser_next_token(p);
    if (p->current_token.type != TOKEN_ASSIGN)
        error(p->current_token.line, p->current_token.col, "預期為 '='");
    parser_next_token(p);
    node->right = parse_expression(p, 0);
    if (p->peek_token.type == TOKEN_SEMI)
        parser_next_token(p);
    return node;
}

// ======================= NEW FUNCTION =======================

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

// ======================= MODIFIED FUNCTION =======================
ASTNode *parse_statement(Parser *p)
{
    switch (p->current_token.type)
    {
    case TOKEN_LET:
        return parse_let_statement(p);
    case TOKEN_RETURN:
        return parse_return_statement(p);
    case TOKEN_IF:
        return parse_if_statement(p); // NEW
    case TOKEN_WHILE:
        return parse_while_statement(p); // NEW
    case TOKEN_LBRACE:
        return parse_block_statement(p);
    default:
        error(p->current_token.line, p->current_token.col, "無效的語句起始符號");
        return NULL;
    }
}

ASTNode *parse_block_statement(Parser *p)
{
    ASTNode *block = create_ast_node(NODE_TYPE_BLOCK);
    block->statements = (ASTNode **)malloc(sizeof(ASTNode *) * 100);
    block->statement_count = 0;
    parser_next_token(p);
    while (p->current_token.type != TOKEN_RBRACE && p->current_token.type != TOKEN_EOF)
    {
        block->statements[block->statement_count++] = parse_statement(p);
        parser_next_token(p);
    }
    if (p->current_token.type != TOKEN_RBRACE)
        error(p->current_token.line, p->current_token.col, "區塊語句缺少 '}'");
    return block;
}
ASTNode *parse_function_declaration(Parser *p)
{
    ASTNode *node = create_ast_node(NODE_TYPE_FUNC_DECL);
    parser_next_token(p);
    if (p->current_token.type != TOKEN_IDENTIFIER)
        error(p->current_token.line, p->current_token.col, "函數宣告缺少名稱");
    node->left = create_ast_node(NODE_TYPE_IDENTIFIER);
    node->left->token = p->current_token;
    parser_next_token(p);
    if (p->current_token.type != TOKEN_LPAREN)
        error(p->current_token.line, p->current_token.col, "函數宣告缺少 '('");
    parser_next_token(p);
    node->params = malloc(sizeof(ASTNode *) * 10);
    node->param_count = 0;
    if (p->current_token.type != TOKEN_RPAREN)
    {
        do
        {
            if (p->current_token.type == TOKEN_COMMA)
                parser_next_token(p);
            if (p->current_token.type != TOKEN_IDENTIFIER)
                error(p->current_token.line, p->current_token.col, "預期為參數名稱");
            ASTNode *param = create_ast_node(NODE_TYPE_IDENTIFIER);
            param->token = p->current_token;
            node->params[node->param_count++] = param;
            parser_next_token(p);
        } while (p->current_token.type == TOKEN_COMMA);
    }
    if (p->current_token.type != TOKEN_RPAREN)
        error(p->current_token.line, p->current_token.col, "函數參數列表缺少 ')'");
    parser_next_token(p);
    if (p->current_token.type != TOKEN_LBRACE)
        error(p->current_token.line, p->current_token.col, "函數主體缺少 '{'");
    node->body = parse_block_statement(p);
    return node;
}
ASTNode *parse_toplevel_declaration(Parser *p)
{
    if (p->current_token.type == TOKEN_FN)
    {
        return parse_function_declaration(p);
    }
    error(p->current_token.line, p->current_token.col, "只允許在頂層宣告函數");
    return NULL;
}
ASTNode *parse_program(Parser *p)
{
    ASTNode *program = create_ast_node(NODE_TYPE_PROGRAM);
    program->statements = (ASTNode **)malloc(sizeof(ASTNode *) * 100);
    program->statement_count = 0;
    while (p->current_token.type != TOKEN_EOF)
    {
        program->statements[program->statement_count++] = parse_toplevel_declaration(p);
        parser_next_token(p);
    }
    return program;
}

//======================================================================
// 5. 中間碼生成器 (IR Generator) - *** 新增 IF/WHILE IR生成 ***
//======================================================================
typedef struct
{
    char name[100];
} Symbol;
typedef struct
{
    Symbol symbols[100];
    int count;
} SymbolTable;
SymbolTable global_symbols;
void add_global_symbol(const char *name) { strcpy(global_symbols.symbols[global_symbols.count++].name, name); }
int find_local_symbol(SymbolTable *table, const char *name)
{
    for (int i = 0; i < table->count; i++)
        if (strcmp(table->symbols[i].name, name) == 0)
            return i;
    return -1;
}
char *generate_ir(ASTNode *node, SymbolTable *local_symbols);
void generate_ir_for_function(ASTNode *node)
{
    emit(OP_FUNC_BEGIN, node->left->token.literal, NULL, NULL);
    SymbolTable local_syms = {.count = 0};
    for (int i = 0; i < node->param_count; i++)
    {
        strcpy(local_syms.symbols[local_syms.count++].name, node->params[i]->token.literal);
    }
    generate_ir(node->body, &local_syms);
    emit(OP_FUNC_END, NULL, NULL, NULL);
}

// ======================= MODIFIED FUNCTION =======================
char *generate_ir(ASTNode *node, SymbolTable *local_symbols)
{
    if (!node)
        return NULL;
    char *left_reg = NULL, *right_reg = NULL, *cond_reg = NULL;
    switch (node->type)
    {
    case NODE_TYPE_PROGRAM:
        for (int i = 0; i < node->statement_count; ++i)
        {
            add_global_symbol(node->statements[i]->left->token.literal);
        }
        for (int i = 0; i < node->statement_count; ++i)
        {
            generate_ir_for_function(node->statements[i]);
        }
        break;
    case NODE_TYPE_BLOCK:
        for (int i = 0; i < node->statement_count; ++i)
        {
            free(generate_ir(node->statements[i], local_symbols));
        }
        break;
    case NODE_TYPE_NUM:
    {
        char *temp = new_temp();
        emit(OP_LOAD_CONST, temp, node->token.literal, NULL);
        return temp;
    }
    case NODE_TYPE_IDENTIFIER:
    {
        if (find_local_symbol(local_symbols, node->token.literal) == -1)
            error(node->token.line, node->token.col, "未宣告的區域變數");
        char *temp = new_temp();
        emit(OP_LOAD_VAR, temp, node->token.literal, NULL);
        return temp;
    }
    case NODE_TYPE_VAR_DECL:
        strcpy(local_symbols->symbols[local_symbols->count++].name, node->left->token.literal);
        right_reg = generate_ir(node->right, local_symbols);
        emit(OP_STORE_VAR, node->left->token.literal, right_reg, NULL);
        free(right_reg);
        break;
    case NODE_TYPE_CALL:
    {
        for (int i = 0; i < node->argument_count; i++)
        {
            char *arg_reg = generate_ir(node->arguments[i], local_symbols);
            emit(OP_ARG, NULL, arg_reg, NULL);
            free(arg_reg);
        }
        char arg_count_str[5];
        snprintf(arg_count_str, 5, "%d", node->argument_count);
        emit(OP_CALL, NULL, node->left->token.literal, arg_count_str);
        char *result_reg = new_temp();
        emit(OP_GET_RETVAL, result_reg, NULL, NULL);
        return result_reg;
    }
    case NODE_TYPE_RETURN:
        right_reg = generate_ir(node->left, local_symbols);
        emit(OP_RETURN, NULL, right_reg, NULL);
        free(right_reg);
        break;
    case NODE_TYPE_BIN_OP:
    {
        left_reg = generate_ir(node->left, local_symbols);
        right_reg = generate_ir(node->right, local_symbols);
        char *result_reg = new_temp();
        OpCode op;
        switch (node->token.type)
        {
        case TOKEN_PLUS:
            op = OP_ADD;
            break;
        case TOKEN_MINUS:
            op = OP_SUB;
            break;
        case TOKEN_MUL:
            op = OP_MUL;
            break;
        case TOKEN_DIV:
            op = OP_DIV;
            break;
        case TOKEN_EQ:
            op = OP_EQ;
            break;
        case TOKEN_NE:
            op = OP_NE;
            break;
        case TOKEN_LT:
            op = OP_LT;
            break;
        case TOKEN_GT:
            op = OP_GT;
            break;
        default:
            error(node->token.line, node->token.col, "未知的二元運算子");
            return NULL;
        }
        emit(op, result_reg, left_reg, right_reg);
        free(left_reg);
        free(right_reg);
        return result_reg;
    }
    // ======================= NEW CASE =======================
    case NODE_TYPE_IF:
    {
        char *else_label = new_label();
        char *end_label = new_label();
        cond_reg = generate_ir(node->condition, local_symbols);
        emit(OP_IF_FALSE_GOTO, NULL, cond_reg, else_label);
        free(cond_reg);
        generate_ir(node->consequence, local_symbols);
        if (node->alternative)
        {
            emit(OP_GOTO, end_label, NULL, NULL);
        }
        emit(OP_LABEL, else_label, NULL, NULL);
        if (node->alternative)
        {
            generate_ir(node->alternative, local_symbols);
        }
        emit(OP_LABEL, end_label, NULL, NULL);
        free(else_label);
        free(end_label);
        return NULL;
    }
    // ======================= NEW CASE =======================
    case NODE_TYPE_WHILE:
    {
        char *start_label = new_label();
        char *end_label = new_label();
        emit(OP_LABEL, start_label, NULL, NULL);
        cond_reg = generate_ir(node->condition, local_symbols);
        emit(OP_IF_FALSE_GOTO, NULL, cond_reg, end_label);
        free(cond_reg);
        generate_ir(node->consequence, local_symbols);
        emit(OP_GOTO, start_label, NULL, NULL);
        emit(OP_LABEL, end_label, NULL, NULL);
        free(start_label);
        free(end_label);
        return NULL;
    }
    default:
        break;
    }
    return NULL;
}

//======================================================================
// 6. 主程式 (輸出檔案)
//======================================================================
void write_ir_to_file(FILE *f, IR_Instruction *code, int count)
{
    for (int i = 0; i < count; i++)
    {
        const char *opcode_str = opcode_to_string(code[i].opcode);
        fprintf(f, "%s %s %s %s\n", opcode_str, strlen(code[i].result) > 0 ? code[i].result : "_", strlen(code[i].arg1) > 0 ? code[i].arg1 : "_", strlen(code[i].arg2) > 0 ? code[i].arg2 : "_");
    }
}
void write_func_info_to_file(FILE *f, ASTNode *ast)
{
    for (int i = 0; i < ast->statement_count; i++)
    {
        ASTNode *func_node = ast->statements[i];
        if (func_node->type == NODE_TYPE_FUNC_DECL)
        {
            fprintf(f, "FUNC_INFO %s %d", func_node->left->token.literal, func_node->param_count);
            for (int j = 0; j < func_node->param_count; j++)
            {
                fprintf(f, " %s", func_node->params[j]->token.literal);
            }
            fprintf(f, "\n");
        }
    }
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        fprintf(stderr, "用法: %s <source_file> <ir_file>\n", argv[0]);
        return 1;
    }
    const char *input_filename = argv[1];
    char *source_code = NULL;
    FILE *source_file = fopen(input_filename, "r");
    if (!source_file)
    {
        perror("無法開啟來源檔案");
        return 1;
    }
    fseek(source_file, 0, SEEK_END);
    long file_size = ftell(source_file);
    rewind(source_file);
    source_code = (char *)malloc(file_size + 1);
    if (!source_code)
    {
        fprintf(stderr, "無法分配記憶體來讀取檔案\n");
        fclose(source_file);
        return 1;
    }
    size_t read_size = fread(source_code, 1, file_size, source_file);
    if ((long)read_size != file_size)
    {
        fprintf(stderr, "讀取檔案時發生錯誤\n");
        free(source_code);
        fclose(source_file);
        return 1;
    }
    source_code[file_size] = '\0';
    fclose(source_file);
    printf("--- 編譯階段 ---\n");
    printf("從檔案 '%s' 讀取原始碼...\n", input_filename);
    global_symbols.count = 0;
    Lexer *lexer = create_lexer(source_code);
    Parser *parser = create_parser(lexer);
    ASTNode *ast = parse_program(parser);
    generate_ir(ast, NULL);
    const char *output_filename = argv[2];
    FILE *out_file = fopen(output_filename, "w");
    if (!out_file)
    {
        perror("無法開啟輸出檔案");
        free(source_code);
        return 1;
    }
    write_func_info_to_file(out_file, ast);
    fprintf(out_file, "CODE_START\n");
    write_ir_to_file(out_file, ir_code, ir_count);
    fclose(out_file);
    printf("\n中間碼已成功寫入到 %s\n", output_filename);
    free(source_code);
    return 0;
}