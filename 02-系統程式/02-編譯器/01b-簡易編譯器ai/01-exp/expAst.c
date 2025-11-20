#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

// --- 權杖定義 ---
typedef enum {
    TOKEN_INTEGER, TOKEN_PLUS, TOKEN_MINUS, TOKEN_MUL, TOKEN_DIV,
    TOKEN_LPAREN, TOKEN_RPAREN, TOKEN_EOF
} TokenType;

typedef struct {
    TokenType type;
    int value;
} Token;

// --- AST 節點定義 ---
typedef enum { NODE_TYPE_NUM, NODE_TYPE_BIN_OP } NodeType;

typedef struct ASTNode {
    NodeType type;
    struct ASTNode *left;
    struct ASTNode *right;
    Token token;
    int value;
} ASTNode;

ASTNode* create_node(NodeType type, Token token, ASTNode* left, ASTNode* right) {
    ASTNode* node = (ASTNode*)malloc(sizeof(ASTNode));
    node->type = type;
    node->token = token;
    node->left = left;
    node->right = right;
    if (type == NODE_TYPE_NUM) {
        node->value = token.value;
    }
    return node;
}

// --- 詞法分析器 ---
typedef struct {
    const char* text;
    int pos;
    Token current_token;
} Lexer;

void error(const char* message) {
    fprintf(stderr, "錯誤: %s\n", message);
    exit(1);
}

Lexer* create_lexer(const char* text) {
    Lexer* lexer = (Lexer*)malloc(sizeof(Lexer));
    lexer->text = text;
    lexer->pos = 0;
    return lexer;
}

void get_next_token(Lexer* lexer) {
    while (lexer->text[lexer->pos] != '\0') {
        if (isspace(lexer->text[lexer->pos])) {
            lexer->pos++;
            continue;
        }
        if (isdigit(lexer->text[lexer->pos])) {
            int value = 0;
            while (isdigit(lexer->text[lexer->pos])) {
                value = value * 10 + (lexer->text[lexer->pos] - '0');
                lexer->pos++;
            }
            lexer->current_token.type = TOKEN_INTEGER;
            lexer->current_token.value = value;
            return;
        }
        char current_char = lexer->text[lexer->pos];
        switch (current_char) {
            case '+': lexer->current_token.type = TOKEN_PLUS; break;
            case '-': lexer->current_token.type = TOKEN_MINUS; break;
            case '*': lexer->current_token.type = TOKEN_MUL; break;
            case '/': lexer->current_token.type = TOKEN_DIV; break;
            case '(': lexer->current_token.type = TOKEN_LPAREN; break;
            case ')': lexer->current_token.type = TOKEN_RPAREN; break;
            default: error("無效字元");
        }
        lexer->pos++;
        return;
    }
    lexer->current_token.type = TOKEN_EOF;
}

// --- 語法分析器 ---
ASTNode* expr(Lexer* lexer); // 前向宣告

ASTNode* factor(Lexer* lexer) {
    Token token = lexer->current_token;
    if (token.type == TOKEN_INTEGER) {
        get_next_token(lexer);
        return create_node(NODE_TYPE_NUM, token, NULL, NULL);
    } else if (token.type == TOKEN_LPAREN) {
        get_next_token(lexer);
        ASTNode* node = expr(lexer);
        if (lexer->current_token.type != TOKEN_RPAREN) {
            error("預期為 ')'");
        }
        get_next_token(lexer);
        return node;
    }
    error("語法錯誤");
    return NULL;
}

ASTNode* term(Lexer* lexer) {
    ASTNode* node = factor(lexer);
    while (lexer->current_token.type == TOKEN_MUL || lexer->current_token.type == TOKEN_DIV) {
        Token op_token = lexer->current_token;
        get_next_token(lexer);
        node = create_node(NODE_TYPE_BIN_OP, op_token, node, factor(lexer));
    }
    return node;
}

ASTNode* expr(Lexer* lexer) {
    ASTNode* node = term(lexer);
    while (lexer->current_token.type == TOKEN_PLUS || lexer->current_token.type == TOKEN_MINUS) {
        Token op_token = lexer->current_token;
        get_next_token(lexer);
        node = create_node(NODE_TYPE_BIN_OP, op_token, node, term(lexer));
    }
    return node;
}

ASTNode* parse(Lexer* lexer) {
    get_next_token(lexer);
    return expr(lexer);
}

// --- 評估器 ---
int evaluate(ASTNode* node) {
    if (!node) return 0;
    if (node->type == NODE_TYPE_NUM) {
        return node->value;
    } else if (node->type == NODE_TYPE_BIN_OP) {
        int left = evaluate(node->left);
        int right = evaluate(node->right);
        switch (node->token.type) {
            case TOKEN_PLUS: return left + right;
            case TOKEN_MINUS: return left - right;
            case TOKEN_MUL: return left * right;
            case TOKEN_DIV:
                if (right == 0) error("除以零");
                return left / right;
            default: error("無效的運算子");
        }
    }
    return 0;
}

// --- 主函式 ---
int main() {
    char input[100];
    printf("請輸入運算式: ");
    fgets(input, 100, stdin);

    Lexer* lexer = create_lexer(input);
    ASTNode* ast = parse(lexer);
    int result = evaluate(ast);

    printf("結果: %d\n", result);

    // 釋放記憶體 (在更複雜的應用中很重要)
    free(lexer);
    // ... 遞迴釋放 AST 節點 ...

    return 0;
}