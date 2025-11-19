// parser.c

#include "sq0.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

// --- 輔助函式原型 (Helpers) ---

// 檢查命令是否為內部元命令 (Meta-Command)，例如 .exit
PrepareResult prepare_meta_command(char *input_buffer);

// 檢查字串長度是否超過限制
PrepareResult check_string_length(const char *token, size_t max_len);

// 將字串中的 ID 轉換為 uint32_t，並檢查 ID 是否為正數
PrepareResult parse_id(const char *token, uint32_t *id);


// --- 輔助函式實作 (Helpers Implementation) ---

PrepareResult prepare_meta_command(char *input_buffer) {
    if (strcmp(input_buffer, ".exit") == 0) {
        printf("Exiting database...\n");
        exit(EXIT_SUCCESS);
    }
    if (strcmp(input_buffer, ".help") == 0) {
        printf("Available meta-commands:\n");
        printf(".exit - Terminate the program.\n");
        printf(".help - Display this help message.\n");
        return PREPARE_SUCCESS; // 成功處理，但不需要執行器介入
    }
    
    // 檢查是否為其他未知的元命令
    if (input_buffer[0] == '.') {
        return PREPARE_UNRECOGNIZED_STATEMENT;
    }

    // 這裡通常不會到達，因為在 prepare_statement 中已經檢查過 input_buffer[0]
    return PREPARE_UNRECOGNIZED_STATEMENT; 
}

PrepareResult check_string_length(const char *token, size_t max_len) {
    if (strlen(token) > max_len) {
        return PREPARE_SYNTAX_ERROR; // 字串過長
    }
    return PREPARE_SUCCESS;
}

PrepareResult parse_id(const char *token, uint32_t *id) {
    char *endptr;
    // 將字串轉換為長整數
    long val = strtol(token, &endptr, 10);

    // 1. 檢查整個 token 是否都轉換成功 (endptr 應該指向 '\0')
    // 2. 檢查 ID 是否為正數 (SQLite ID 通常從 1 開始)
    if (*endptr != '\0' || val <= 0) {
        return PREPARE_SYNTAX_ERROR;
    }
    
    *id = (uint32_t)val;
    return PREPARE_SUCCESS;
}


// --- 核心函式實作 (Core Implementation) ---

/**
 * @brief 接收使用者輸入，檢查語法，並準備 Statement 結構。
 * @param input_buffer 使用者輸入的字串。
 * @param statement 指向要填充的 Statement 結構。
 * @return PrepareResult 準備結果狀態碼。
 */
PrepareResult prepare_statement(char *input_buffer, Statement *statement) {
    // 1. 處理元命令 (Meta-commands)
    if (input_buffer[0] == '.') {
        return prepare_meta_command(input_buffer);
    }

    // 2. 處理 SELECT 語句
    // 我們使用 strncmp 檢查關鍵字，然後檢查後面是否沒有其他字符
    if (strncmp(input_buffer, "select", 6) == 0 && 
        (input_buffer[6] == '\0' || isspace((unsigned char)input_buffer[6]))) {
        
        // 簡易版只處理單詞 "select" 
        if (strlen(input_buffer) == 6) { 
            statement->type = STATEMENT_SELECT;
            return PREPARE_SUCCESS;
        }
    }

    // 3. 處理 INSERT 語句 (語法: insert ID USERNAME EMAIL)
    if (strncmp(input_buffer, "insert", 6) == 0 && isspace((unsigned char)input_buffer[6])) {
        statement->type = STATEMENT_INSERT;

        char *token;
        char *saveptr; // 用於 strtok_r 的內部狀態

        // 複製一份輸入字串，因為 strtok_r 會修改原始字串
        char *buffer_copy = strdup(input_buffer); 
        if (!buffer_copy) {
            perror("strdup failed");
            exit(EXIT_FAILURE);
        }
        
        // 為了確保釋放，使用 goto 標籤處理錯誤
        PrepareResult result = PREPARE_SUCCESS;

        // 第一個 token 應該是 "insert" (跳過)
        token = strtok_r(buffer_copy, " ", &saveptr); 
        
        // --- ID (第二個 token) ---
        token = strtok_r(NULL, " ", &saveptr);
        if (!token || parse_id(token, &statement->row_to_insert.id) != PREPARE_SUCCESS) {
             result = PREPARE_SYNTAX_ERROR;
             goto cleanup;
        }

        // --- USERNAME (第三個 token) ---
        token = strtok_r(NULL, " ", &saveptr);
        // 假設 Row 結構的 username 大小為 32
        if (!token || check_string_length(token, 31) != PREPARE_SUCCESS) { 
             result = PREPARE_SYNTAX_ERROR;
             goto cleanup;
        }
        // 複製字串到 Row 結構中
        strcpy(statement->row_to_insert.username, token);

        // --- EMAIL (第四個 token) ---
        token = strtok_r(NULL, " ", &saveptr);
        // 假設 Row 結構的 email 大小為 255
        if (!token || check_string_length(token, 254) != PREPARE_SUCCESS) { 
             result = PREPARE_SYNTAX_ERROR;
             goto cleanup;
        }
        strcpy(statement->row_to_insert.email, token);

        // --- 檢查是否有額外的 Token (語法錯誤) ---
        token = strtok_r(NULL, " ", &saveptr);
        if (token != NULL) {
            result = PREPARE_SYNTAX_ERROR; // 語句中有多餘的內容
            goto cleanup;
        }

cleanup:
        free(buffer_copy);
        return result;
    }

    // 4. 其他未識別的語句
    return PREPARE_UNRECOGNIZED_STATEMENT;
}
