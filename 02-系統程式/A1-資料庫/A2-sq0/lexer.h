// lexer.h

#ifndef LEXER_H
#define LEXER_H

#include <stdint.h>
#include <stdlib.h> // 需要 size_t

// --- I. Token 類型 ---
typedef enum {
    TOKEN_EOF,              // 輸入結尾 (End of File / End of Input)
    TOKEN_ILLEGAL,          // 無法識別的字元或序列
    
    // 關鍵字 (Keywords)
    TOKEN_KEYWORD_INSERT, 
    TOKEN_KEYWORD_SELECT, 
    TOKEN_KEYWORD_EXIT,     // 元命令
    TOKEN_KEYWORD_HELP,     // 元命令

    // 識別符號與字面值 (Identifiers and Literals)
    TOKEN_IDENTIFIER,       // 識別符號 (如：欄位名, 表格名)
    TOKEN_INTEGER,          // 數字 (如：ID 值, 1, 2, 3)
    TOKEN_STRING,           // 字串 (如：'user@example.com')
    
    // 符號 (Symbols)
    TOKEN_SEMICOLON,        // 分號 (;)
    // ... 可擴充如 COMMA (,), LPAREN ((), RPAREN ()) 等
    
} TokenType;

// --- II. Token 結構 ---
typedef struct {
    TokenType type;
    char *start;    // Token 在原始輸入字串中的起始位置
    int length;     // Token 的長度
} Token;

// --- III. 詞法分析器狀態 ---
typedef struct {
    char *input;    // 原始輸入字串
    size_t length;  // 輸入字串的總長度
    int current_pos; // 當前在輸入字串中的位置
} Lexer;


// --- IV. 函式介面 (Function Prototypes) ---

/**
 * @brief 初始化 Lexer 狀態。
 * @param input 原始輸入字串。
 * @return Lexer* 初始化後的 Lexer 指標。
 */
Lexer *lexer_init(char *input);

/**
 * @brief 獲取下一個 Token。
 * @param lexer 詞法分析器狀態。
 * @return Token 結構。
 */
Token lexer_next_token(Lexer *lexer);

/**
 * @brief 釋放 Lexer 資源。
 * @param lexer 要釋放的 Lexer 指標。
 */
void lexer_free(Lexer *lexer);

#endif // LEXER_H
