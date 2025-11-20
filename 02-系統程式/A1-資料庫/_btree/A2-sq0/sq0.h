// sq0.h

#ifndef SQ0_H
#define SQ0_H

#include <stdint.h>
#include <stdbool.h>

// --- 常數定義 ---
#define PAGE_SIZE 4096      // 資料庫頁面大小 (4KB)
#define TABLE_MAX_PAGES 100 // 簡化：資料庫最大頁數

// --- 語句類型 ---
typedef enum {
    STATEMENT_INSERT,
    STATEMENT_SELECT,
    // ... 可擴充如 DELETE, UPDATE, CREATE TABLE
} StatementType;

// --- 預處理結果 ---
typedef enum {
    PREPARE_SUCCESS,
    PREPARE_UNRECOGNIZED_STATEMENT,
    PREPARE_SYNTAX_ERROR,
    // ...
} PrepareResult;

// --- 執行結果 ---
typedef enum {
    EXECUTE_SUCCESS,
    EXECUTE_TABLE_FULL, // 簡易版的常見錯誤
    // ...
} ExecuteResult;


// --- 核心結構定義 ---

// 用來表示一行資料 (簡化為 id, username, email)
typedef struct {
    uint32_t id;
    char username[32];
    char email[255];
} Row;

// 表示一個 SQL 語句 (準備好的操作)
typedef struct {
    StatementType type;
    Row row_to_insert; // 僅用於 INSERT 語句
} Statement;

// 資料庫連接 (包含 Pager 和其他資訊)
typedef struct {
    // Pager *pager; // 這裡應該包含 Pager 結構
    // Table *table; // 這裡應該包含 Table 結構
    void *connection_data; // 抽象化，用於指向更複雜的結構
} DbConnection;


// --- 函式介面 (Function Prototypes) ---

// 1. 預處理 (Parser)
PrepareResult prepare_statement(char *input_buffer, Statement *statement);

// 2. 執行 (VM/Executor)
ExecuteResult execute_statement(Statement *statement, DbConnection *conn);

// 3. 連線管理
DbConnection *db_open(const char *filename);
void db_close(DbConnection *conn);

#endif // SQ0_H