// vm.c

#include "sq0.h"
#include "btree.h" // 執行器需要操作 Table 和 Cursor
#include "util.h"  // 執行器需要反序列化 Row
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


// --- 輔助函式 (Helper Functions) ---

/**
 * @brief 打印一行資料。
 * @param row 指向要打印的 Row 結構。
 */
void print_row(Row *row) {
    printf("(%u, %s, %s)\n", row->id, row->username, row->email);
}


// --- 執行 INSERT 語句 ---

/**
 * @brief 執行 INSERT 語句，將資料寫入 B-Tree。
 * * @param statement 包含要插入的 Row 資料。
 * @param table 指向目標 Table 結構。
 * @return ExecuteResult 執行結果。
 */
ExecuteResult execute_insert(Statement *statement, Table *table) {
    // 1. 定位插入點：找到 Key (ID) 應該被插入的位置
    Cursor *cursor = table_find(table, statement->row_to_insert.id);

    // 檢查 Key 是否已經存在 (簡易衝突檢測)
    void *node = get_page(table->pager, cursor->page_num);
    uint32_t num_cells = *leaf_node_num_cells(node);

    if (cursor->cell_num < num_cells) {
        uint32_t key_at_index = *leaf_node_key(node, cursor->cell_num);
        if (key_at_index == statement->row_to_insert.id) {
            // 如果游標指向的單元格的 Key 與要插入的 Key 相同，則表示發生衝突
            printf("Error: Key '%u' already exists.\n", statement->row_to_insert.id);
            free(cursor);
            return EXECUTE_SUCCESS; // 雖然是錯誤，但我們讓程式保持運行
        }
    }

    // 2. 執行 B-Tree 插入操作
    leaf_node_insert(cursor, statement->row_to_insert.id, &statement->row_to_insert);

    free(cursor);
    return EXECUTE_SUCCESS;
}


// --- 執行 SELECT 語句 ---

/**
 * @brief 執行 SELECT 語句，遍歷 B-Tree 並打印所有資料。
 * * @param statement 包含要執行的 SELECT 語句資訊。
 * @param table 指向目標 Table 結構。
 * @return ExecuteResult 執行結果。
 */
ExecuteResult execute_select(Statement *statement, Table *table) {
    (void)statement; // 未使用參數，避免編譯警告
    Cursor *cursor = table_start(table); // 獲取從 B-Tree 起點開始的游標
    Row row; // 用於存放反序列化後的 Row 資料

    printf("\n--- Results ---\n");
    // 遍歷所有節點，直到游標到達表格末端
    while (!(cursor->end_of_table)) {
        // 1. 獲取當前游標所在的頁面和單元格數據
        void *node = get_page(table->pager, cursor->page_num);
        void *value_ptr = leaf_node_value(node, cursor->cell_num);
        
        // 2. 反序列化數據到 Row 結構
        deserialize_row(value_ptr, &row);
        
        // 3. 打印結果
        print_row(&row);
        
        // 4. 移動到下一個單元格
        cursor_advance(cursor);
    }

    printf("---------------\n");
    free(cursor);
    return EXECUTE_SUCCESS;
}


// --- 核心執行器函式 ---

/**
 * @brief 根據 Statement 類型，呼叫對應的執行函式。
 * * @param statement 準備好的 Statement 結構。
 * @param conn DbConnection 結構 (我們使用 conn->connection_data 作為 Table 指標)。
 * @return ExecuteResult 執行結果狀態碼。
 */
ExecuteResult execute_statement(Statement *statement, DbConnection *conn) {
    // 假設 DbConnection->connection_data 指向 Table 結構
    Table *table = (Table *)conn->connection_data;
    
    switch (statement->type) {
        case STATEMENT_INSERT:
            return execute_insert(statement, table);
            
        case STATEMENT_SELECT:
            return execute_select(statement, table);
            
        default:
            fprintf(stderr, "VM Error: Unhandled statement type.\n");
            return EXECUTE_SUCCESS; // 這裡也可以定義 EXECUTE_ERROR
    }
}
