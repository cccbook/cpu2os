// lexer.c

#include "lexer.h"
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdio.h>

// --- 關鍵字對照表 ---
// 在實際的 Lexer 中，通常會用 Hash Table 來加速查找，這裡使用簡單的陣列
struct {
    const char *literal;
    TokenType type;
} keywords[] = {
    {"insert", TOKEN_KEYWORD_INSERT},
    {"select", TOKEN_KEYWORD_SELECT},
    {".exit", TOKEN_KEYWORD_EXIT},
    {".help", TOKEN_KEYWORD_HELP},
    // 添加其他關鍵字
};
const int NUM_KEYWORDS = sizeof(keywords) / sizeof(keywords[0]);


// --- 輔助函式 (Helpers) ---

// 跳過空白字元 (空格、tab、換行)
static void skip_whitespace(Lexer *lexer) {
    while (isspace((unsigned char)lexer->input[lexer->current_pos])) {
        lexer->current_pos++;
    }
}

// 檢查並返回關鍵字類型
static TokenType check_keyword(const char *literal, int length) {
    for (int i = 0; i < NUM_KEYWORDS; i++) {
        if (strlen(keywords[i].literal) == (size_t)length && 
            strncmp(keywords[i].literal, literal, length) == 0) {
            return keywords[i].type;
        }
    }
    // 如果不是關鍵字，就是一般的識別符號
    return TOKEN_IDENTIFIER;
}


// --- 核心函式實作 ---

Lexer *lexer_init(char *input) {
    Lexer *lexer = (Lexer *)malloc(sizeof(Lexer));
    if (!lexer) {
        perror("Failed to allocate lexer");
        exit(EXIT_FAILURE);
    }
    lexer->input = input;
    lexer->current_pos = 0;
    return lexer;
}

void lexer_free(Lexer *lexer) {
    // 這裡只釋放 Lexer 結構本身，不釋放 input 字串 (因為它是從 main 傳入的)
    free(lexer);
}

/**
 * @brief 獲取下一個 Token
 * @param lexer 詞法分析器狀態
 * @return Token 結構
 */
Token lexer_next_token(Lexer *lexer) {
    skip_whitespace(lexer);

    char current_char = lexer->input[lexer->current_pos];
    Token token;
    token.start = &lexer->input[lexer->current_pos];
    token.length = 0;

    if (current_char == '\0') {
        // 達到輸入結尾
        token.type = TOKEN_EOF;
        return token;
    }

    // 1. 處理符號 (Symbol)
    switch (current_char) {
        case ';':
            token.type = TOKEN_SEMICOLON;
            token.length = 1;
            lexer->current_pos++;
            return token;
        // ... 可擴充如 + - * / 等
    }

    // 2. 處理數字 (Integer)
    if (isdigit((unsigned char)current_char)) {
        int start_pos = lexer->current_pos;
        while (isdigit((unsigned char)lexer->input[lexer->current_pos])) {
            lexer->current_pos++;
        }
        token.type = TOKEN_INTEGER;
        token.length = lexer->current_pos - start_pos;
        return token;
    }

    // 3. 處理識別符號和關鍵字 (Identifier/Keyword)
    if (isalpha((unsigned char)current_char) || current_char == '.') {
        int start_pos = lexer->current_pos;
        lexer->current_pos++; // 處理第一個字母或 '.'
        
        while (isalnum((unsigned char)lexer->input[lexer->current_pos]) || lexer->input[lexer->current_pos] == '_') {
            lexer->current_pos++;
        }
        
        token.length = lexer->current_pos - start_pos;
        // 檢查是否為關鍵字
        token.type = check_keyword(token.start, token.length);
        
        return token;
    }

    // 4. 處理字串 (String) - 簡化：這裡我們暫時不處理單引號括起來的字串，
    // 而是將 email/username 視為一般的 Identifier 處理。
    // 如果需要支援 'username'，則需要添加如下邏輯：
    /*
    if (current_char == '\'') {
        lexer->current_pos++;
        int start_pos = lexer->current_pos;
        while (lexer->input[lexer->current_pos] != '\0' && lexer->input[lexer->current_pos] != '\'') {
             lexer->current_pos++;
        }
        if (lexer->input[lexer->current_pos] == '\'') {
            token.type = TOKEN_STRING;
            token.start = &lexer->input[start_pos];
            token.length = lexer->current_pos - start_pos;
            lexer->current_pos++; // 跳過結尾的 '
            return token;
        }
        // 如果沒有匹配到結尾的單引號，這是一個詞法錯誤
    }
    */


    // 5. 無法識別的字元
    fprintf(stderr, "Lexer Error: Unrecognized character at position %d: '%c'\n", 
            lexer->current_pos, current_char);
    // 這裡可以選擇返回一個錯誤 Token 或跳過
    lexer->current_pos++; 
    token.type = TOKEN_EOF; // 簡單處理，讓 Parser 停下來
    return token;
}
