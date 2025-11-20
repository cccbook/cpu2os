// btree.h

#ifndef BTREE_H
#define BTREE_H

#include <stdint.h>
#include <stdbool.h>
#include "sq0.h"
#include "pager.h"
#include "util.h" // 需要 ROW_SIZE

// --- I. B-Tree 節點類型 ---
typedef enum {
    NODE_INTERNAL, // 內部節點：儲存 Key 和指向子節點的頁碼
    NODE_LEAF      // 葉節點：儲存 Key 和實際的資料 (Row)
} NodeType;


// --- II. 節點佈局常量 (Node Layout Constants) ---

// 1. 通用節點頭部 (Common Node Header) - 適用於所有節點
static const uint32_t NODE_TYPE_SIZE = sizeof(uint8_t);
static const uint32_t IS_ROOT_SIZE = sizeof(uint8_t); // 標記是否為根節點
static const uint32_t PARENT_POINTER_SIZE = sizeof(uint32_t); // 父節點的頁碼

static const uint32_t COMMON_NODE_HEADER_SIZE = 
    NODE_TYPE_SIZE + IS_ROOT_SIZE + PARENT_POINTER_SIZE;

// 2. 葉節點頭部 (Leaf Node Header)
static const uint32_t LEAF_NODE_NUM_CELLS_SIZE = sizeof(uint32_t); // 儲存的單元格數量
static const uint32_t LEAF_NODE_HEADER_SIZE = 
    COMMON_NODE_HEADER_SIZE + LEAF_NODE_NUM_CELLS_SIZE;

// 3. 葉節點單元格 (Leaf Node Cell)
static const uint32_t KEY_SIZE = sizeof(uint32_t); // Key (例如：Row ID)
static const uint32_t VALUE_SIZE = ROW_SIZE;       // Value (即序列化後的 Row)
static const uint32_t LEAF_NODE_CELL_SIZE = KEY_SIZE + VALUE_SIZE;

// 4. 葉節點容量計算
static const uint32_t LEAF_NODE_SPACE_FOR_CELLS = 
    PAGE_SIZE - LEAF_NODE_HEADER_SIZE;
static const uint32_t LEAF_NODE_MAX_CELLS = 
    LEAF_NODE_SPACE_FOR_CELLS / LEAF_NODE_CELL_SIZE;


// --- III. 高階結構定義 ---

// 表格 (Table) 結構，將 B-Tree 結構與 Pager 連接起來
typedef struct {
    Pager *pager;
    uint32_t root_page_num; // 根節點所在的頁面編號
} Table;

// 資料指標 (Cursor) 結構，用於 SELECT 操作的迭代或 INSERT 的定位
typedef struct {
    Table *table;
    uint32_t page_num;  // 目前所在的頁面
    uint32_t cell_num;  // 目前在頁面中的單元格編號
    bool end_of_table;  // 是否已經到達表格末端 (用於 SELECT)
} Cursor;


// --- IV. 函式介面 (Function Prototypes) ---

// 1. Table/Connection 函式
Table *db_open_table(const char *filename);
void db_close_table(Table *table);

// 2. Cursor 函式 (遍歷)
Cursor *table_start(Table *table);
Cursor *table_find(Table *table, uint32_t key); // 搜尋特定 Key 的插入/讀取位置
void cursor_advance(Cursor *cursor); 

// 3. B-Tree 節點操作 (供 VM 呼叫)
void leaf_node_insert(Cursor *cursor, uint32_t key, Row *value);
void initialize_leaf_node(void *node);

// 4. 節點存取輔助函式 (Getter/Setter)
// 由於這些函式操作底層記憶體，它們通常是 btree.c 的靜態內部函式，
// 但為了允許像 vm.c 這樣的高層次模組存取數據，我們需要將讀取函式暴露出來。
// * 注意：雖然暴露這些函式違反了某些模組化原則，但在 C 語言小型專案中常見。

// 獲取節點類型
NodeType get_node_type(void *node);

// 獲取葉節點的單元格數量指標 (Pointer to num_cells field)
uint32_t *leaf_node_num_cells(void *node);

// 獲取單元格內的 Key 指標
uint32_t *leaf_node_key(void *node, uint32_t cell_num);

// 獲取單元格內的 Value (Row 資料) 指標
void *leaf_node_value(void *node, uint32_t cell_num);


#endif // BTREE_H
