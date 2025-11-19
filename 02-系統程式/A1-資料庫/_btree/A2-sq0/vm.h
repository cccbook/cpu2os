// vm.h

#ifndef VM_H
#define VM_H

#include "sq0.h"

// --- 函式介面 (Function Prototypes) ---

/**
 * @brief 根據 Statement 類型，呼叫對應的執行函式。
 * 這是資料庫引擎的核心執行器。
 * * @param statement 準備好的 Statement 結構。
 * @param conn DbConnection 結構 (包含 Table/Pager 資訊)。
 * @return ExecuteResult 執行結果狀態碼。
 */
ExecuteResult execute_statement(Statement *statement, DbConnection *conn);

#endif // VM_H