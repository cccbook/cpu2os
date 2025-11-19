// test_sq0.c

#include "sq0.h"
#include "btree.h"
#include "vm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h> // For unlink()

// 測試用的資料庫檔案名
#define TEST_DB_FILENAME "test_sq0.db"
#define MAX_INPUT_LEN 256

// 全域變數，用於追蹤測試狀態
int tests_run = 0;
int tests_failed = 0;

// =================================================================
// 輔助函式 (Helper Functions)
// =================================================================

/**
 * @brief 運行一個 SQL 語句。
 * @param conn 資料庫連線。
 * @param sql_command SQL 語句字串。
 * @return ExecuteResult 執行結果。
 */
ExecuteResult run_sql(DbConnection *conn, const char *sql_command) {
    char command_buffer[MAX_INPUT_LEN];
    Statement statement;
    PrepareResult prep_res;
    ExecuteResult exec_res;

    // 複製命令到可修改的緩衝區
    strncpy(command_buffer, sql_command, MAX_INPUT_LEN - 1);
    command_buffer[MAX_INPUT_LEN - 1] = '\0';

    // 1. 準備 (Parser)
    prep_res = prepare_statement(command_buffer, &statement);

    if (prep_res != PREPARE_SUCCESS) {
        // 如果是元命令或無法識別的語句，也視為成功運行（除非是語法錯誤）
        if (prep_res == PREPARE_UNRECOGNIZED_STATEMENT || 
            prep_res == PREPARE_SYNTAX_ERROR) {
            fprintf(stderr, "Error: Failed to prepare statement '%s' (Code: %d)\n", sql_command, prep_res);
            return -1; // 使用 -1 表示測試失敗的執行結果
        }
        return EXECUTE_SUCCESS;
    }

    // 2. 執行 (VM)
    exec_res = execute_statement(&statement, conn);
    return exec_res;
}

/**
 * @brief 執行 SELECT 並計算結果行數。
 * * 這是測試的核心，它需要模擬 execute_select 的邏輯來計數。
 * @param conn 資料庫連線。
 * @return int 查詢到的行數。
 */
int count_rows(DbConnection *conn) {
    Table *table = (Table *)conn->connection_data;
    Cursor *cursor = table_start(table);
    int count = 0;

    while (!(cursor->end_of_table)) {
        count++;
        cursor_advance(cursor);
    }
    free(cursor);
    return count;
}


#define ASSERT_TRUE(condition, message) \
    do { \
        tests_run++; \
        if (!(condition)) { \
            fprintf(stderr, "Test Failed: %s, Line %d: %s\n", __FILE__, __LINE__, message); \
            tests_failed++; \
        } \
    } while (0)

// =================================================================
// 測試案例 (Test Cases)
// =================================================================

/**
 * @brief 測試插入、查詢和行數計數功能。
 */
void test_insert_and_select() {
    printf("--- Running Test 1: Insert and Select ---\n");
    DbConnection *conn = db_open(TEST_DB_FILENAME);

    // 確保資料庫是空的 (以防測試前有殘留數據)
    ASSERT_TRUE(count_rows(conn) == 0, "Initial row count must be 0");

    // 插入三筆資料
    run_sql(conn, "insert 1 user1 user1@example.com");
    run_sql(conn, "insert 2 user2 user2@example.com");
    run_sql(conn, "insert 100 max_id_user max@id.com");
    
    // 檢查行數是否為 3
    ASSERT_TRUE(count_rows(conn) == 3, "After 3 inserts, row count must be 3");

    // 執行 select (檢查是否成功運行，細節結果需手動檢查終端輸出)
    ExecuteResult res = run_sql(conn, "select");
    ASSERT_TRUE(res == EXECUTE_SUCCESS, "SELECT execution must succeed");

    db_close(conn);
    printf("--- Test 1 Finished ---\n");
}

/**
 * @brief 測試資料持久化功能 (讀取磁碟)。
 */
void test_persistence() {
    printf("--- Running Test 2: Persistence ---\n");
    
    // 在 Test 1 中已經插入了 3 筆資料並關閉了檔案 (db_close會flush)
    
    // 重新開啟資料庫檔案
    DbConnection *conn = db_open(TEST_DB_FILENAME);
    
    // 檢查行數是否仍然是 3
    int count = count_rows(conn);
    ASSERT_TRUE(count == 3, "After reopening, row count must still be 3");

    // 插入一筆新資料
    run_sql(conn, "insert 3 user3 user3@example.com");
    
    // 檢查行數是否變為 4
    ASSERT_TRUE(count_rows(conn) == 4, "After inserting a new row, count must be 4");

    db_close(conn);
    printf("--- Test 2 Finished ---\n");
}

/**
 * @brief 測試重複 Key 插入功能 (Key Collision)。
 */
void test_key_collision() {
    printf("--- Running Test 3: Key Collision ---\n");
    DbConnection *conn = db_open(TEST_DB_FILENAME);
    
    // Test 2 結束時有 4 筆資料
    int initial_count = count_rows(conn); // 應該是 4
    
    // 嘗試插入 Key = 1 的重複資料 (Key 1 已經存在)
    // 預期：執行器會打印錯誤，但 count_rows 保持不變
    run_sql(conn, "insert 1 duplicate_user duplicate@key.com"); 

    // 檢查行數是否保持不變
    int final_count = count_rows(conn);
    ASSERT_TRUE(final_count == initial_count, "Row count must not change on key collision");
    
    db_close(conn);
    printf("--- Test 3 Finished ---\n");
}


// =================================================================
// 主函式
// =================================================================

int main() {
    // 每次測試前，先刪除舊的資料庫檔案
    unlink(TEST_DB_FILENAME); 

    // 運行所有測試
    test_insert_and_select();
    test_persistence();
    test_key_collision();

    printf("\n==================================\n");
    printf("Tests Run: %d, Failed: %d\n", tests_run, tests_failed);
    printf("==================================\n");

    // 再次刪除資料庫檔案，保持環境清潔
    unlink(TEST_DB_FILENAME); 

    return tests_failed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}