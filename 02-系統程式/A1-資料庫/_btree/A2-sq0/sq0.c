#include "sq0.h"
#include "btree.h" // 需要操作 Table 結構
#include "util.h"  // 需要序列化/反序列化 Row
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


// --- 核心連線處理 ---

/**
 * @brief 打開資料庫連線。
 * @param filename 資料庫檔案名稱。
 * @return DbConnection* 連線結構指標。
 */
DbConnection *db_open(const char *filename) {
    // 1. 開啟底層 Table (會自動處理 Pager 和根節點初始化)
    Table *table = db_open_table(filename);
    
    // 2. 初始化 DbConnection 結構
    DbConnection *conn = (DbConnection *)malloc(sizeof(DbConnection));
    if (!conn) {
        perror("Failed to allocate DbConnection");
        exit(EXIT_FAILURE);
    }
    
    // 3. 將 Table 結構存入 connection_data 供執行器使用
    conn->connection_data = table;
    
    return conn;
}

/**
 * @brief 關閉資料庫連線。
 * @param conn 連線結構指標。
 */
void db_close(DbConnection *conn) {
    if (conn) {
        Table *table = (Table *)conn->connection_data;
        if (table) {
            db_close_table(table); // 關閉 Table (會 Flush Pager)
        }
        free(conn);
    }
}

