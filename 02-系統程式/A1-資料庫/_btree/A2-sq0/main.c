// main.c

#include "sq0.h"
#include "btree.h" // 需要 Table 結構
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

// --- REPL 輔助結構 ---

// 定義一個結構來處理使用者輸入緩衝區
typedef struct {
    char *buffer;
    size_t buffer_length;
    ssize_t input_length;
} InputBuffer;

// --- 輔助函式原型 ---
void print_prompt();
InputBuffer *new_input_buffer();
void read_input(InputBuffer *input_buffer);
void close_input_buffer(InputBuffer *input_buffer);

// --- REPL 輔助函式實作 ---

void print_prompt() {
    printf("sq0> ");
}

InputBuffer *new_input_buffer() {
    InputBuffer *input_buffer = (InputBuffer *)malloc(sizeof(InputBuffer));
    if (!input_buffer) {
        perror("Failed to allocate InputBuffer");
        exit(EXIT_FAILURE);
    }
    input_buffer->buffer = NULL;
    input_buffer->buffer_length = 0;
    input_buffer->input_length = 0;
    return input_buffer;
}

void read_input(InputBuffer *input_buffer) {
    // 使用 getline 讀取輸入，自動管理記憶體
    ssize_t bytes_read = getline(&(input_buffer->buffer), &(input_buffer->buffer_length), stdin);
    
    if (bytes_read <= 0) {
        // 處理 EOF 或 I/O 錯誤
        if (feof(stdin)) {
            printf("\nEOF encountered. Exiting.\n");
        } else if (ferror(stdin)) {
            perror("Error reading input");
        }
        exit(EXIT_FAILURE);
    }

    // 去除結尾的換行符號 (Newline)
    input_buffer->input_length = bytes_read - 1;
    input_buffer->buffer[input_buffer->input_length] = '\0';
}

void close_input_buffer(InputBuffer *input_buffer) {
    free(input_buffer->buffer);
    free(input_buffer);
}


// --- 主程式進入點 ---

int main(int argc, char *argv[]) {
    // 檢查是否有提供資料庫檔案名稱
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <db_filename>\n", argv[0]);
        return EXIT_FAILURE;
    }
    
    const char *db_filename = argv[1];
    
    // 1. 初始化連線
    DbConnection *connection = db_open(db_filename);

    // 2. 初始化輸入緩衝區
    InputBuffer *input_buffer = new_input_buffer();
    
    printf("--- sq0 Mini-SQLite Engine Initialized ---\n");
    printf("Database file: %s\n", db_filename);
    printf("Type '.help' for commands or '.exit' to quit.\n");

    // 3. 進入 REPL 迴圈
    while (1) {
        print_prompt();
        read_input(input_buffer);

        // 處理空行
        if (input_buffer->input_length == 0) {
            continue;
        }

        Statement statement;
        PrepareResult prepare_result = prepare_statement(input_buffer->buffer, &statement);

        switch (prepare_result) {
            case PREPARE_SUCCESS:
                // 語句準備成功，交給執行器
                execute_statement(&statement, connection);
                break;

            case PREPARE_UNRECOGNIZED_STATEMENT:
                printf("Error: Unrecognized command '%s'\n", input_buffer->buffer);
                break;
                
            case PREPARE_SYNTAX_ERROR:
                printf("Error: Syntax error or invalid data.\n");
                printf("Expected format: insert ID USERNAME EMAIL\n");
                break;
                
            // 由於 .exit 和 .help 在 parser.c 中處理並直接退出了，
            // 這裡不需要額外處理它們的 PrepareResult。
        }
    }

    // 4. 程式結束時，釋放資源 (雖然通常被 .exit 截斷，但為了健壯性仍保留)
    close_input_buffer(input_buffer);
    db_close(connection);

    return EXIT_SUCCESS;
}
