// btree.c
#include "btree.h"
#include "pager.h"
#include "sq0.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
// --- 輔助函式 (Getter/Setter for Node Fields) ---

// 設定/取得節點類型
void set_node_type(void *node, NodeType type) {
    uint8_t value = type;
    *((uint8_t *)(node)) = value;
}

NodeType get_node_type(void *node) {
    return (NodeType)*((uint8_t *)(node));
}

// 取得葉節點的單元格數量指標 (Pointer to num_cells field)
uint32_t *leaf_node_num_cells(void *node) {
    // 偏移量：通用頭部大小
    return (uint32_t *)(node + COMMON_NODE_HEADER_SIZE);
}

// 取得單元格的記憶體位置
void *leaf_node_cell(void *node, uint32_t cell_num) {
    uint32_t offset = LEAF_NODE_HEADER_SIZE + cell_num * LEAF_NODE_CELL_SIZE;
    return node + offset;
}

// 取得單元格內的 Key 指標
uint32_t *leaf_node_key(void *node, uint32_t cell_num) {
    // Key 在 Cell 的最前面
    return (uint32_t *)leaf_node_cell(node, cell_num);
}

// 取得單元格內的 Value (Row 資料) 指標
void *leaf_node_value(void *node, uint32_t cell_num) {
    // Value 在 Key 的後面
    return leaf_node_cell(node, cell_num) + KEY_SIZE;
}

// 初始化一個葉節點
void initialize_leaf_node(void *node) {
    set_node_type(node, NODE_LEAF);
    // 假設非根節點且無父節點 (簡化)
    // 設置單元格數量為 0
    *leaf_node_num_cells(node) = 0; 
}

// db_open_table 定義於 btree.c
Table *db_open_table(const char *filename) {
    Pager *pager = pager_open(filename);
    
    Table *table = (Table *)malloc(sizeof(Table));
    table->pager = pager;
    table->root_page_num = 0;

    if (pager->num_pages == 0) {
        // 資料庫檔案為空，初始化根節點
        void *root_node = get_page(pager, 0);
        initialize_leaf_node(root_node);
    }

    return table;
}

// db_close_table 定義於 btree.c
void db_close_table(Table *table) {
    Pager *pager = table->pager;
    
    // 寫回所有快取的修改 (Flush dirty pages)
    for (uint32_t i = 0; i < pager->num_pages; i++) {
        if (pager->pages[i] == NULL) {
            continue;
        }
        pager_flush(pager, i);
        free(pager->pages[i]);
        pager->pages[i] = NULL;
    }

    // 關閉 Pager/檔案
    pager_close(pager);
    free(table);
}

// table_start 定義於 btree.c
Cursor *table_start(Table *table) {
    Cursor *cursor = (Cursor *)malloc(sizeof(Cursor));
    cursor->table = table;
    cursor->page_num = table->root_page_num;
    cursor->cell_num = 0;

    void *root_node = get_page(table->pager, table->root_page_num);
    uint32_t num_cells = *leaf_node_num_cells(root_node);

    // 如果根節點沒有單元格，則指標在表格末端
    cursor->end_of_table = (num_cells == 0);

    return cursor;
}

// cursor_advance 定義於 btree.c
void cursor_advance(Cursor *cursor) {
    // 簡化：只處理單一葉節點的 B-Tree
    cursor->cell_num += 1;

    void *node = get_page(cursor->table->pager, cursor->page_num);
    uint32_t num_cells = *leaf_node_num_cells(node);

    if (cursor->cell_num >= num_cells) {
        cursor->end_of_table = true;
    }
}

// table_find 定義於 btree.c
Cursor *table_find(Table *table, uint32_t key) {
    // 簡化：只在根節點（Page 0）上進行二分搜尋
    
    uint32_t root_page_num = table->root_page_num;
    void *node = get_page(table->pager, root_page_num);
    uint32_t num_cells = *leaf_node_num_cells(node);

    Cursor *cursor = (Cursor *)malloc(sizeof(Cursor));
    cursor->table = table;
    cursor->page_num = root_page_num;
    cursor->end_of_table = false;
    
    // 二分搜尋 (Binary Search) 找到 Key 的插入位置
    uint32_t min_index = 0;
    uint32_t max_index = num_cells; 
    
    while (min_index < max_index) {
        uint32_t index = (min_index + max_index) / 2;
        uint32_t cell_key = *leaf_node_key(node, index);
        
        if (key <= cell_key) {
            max_index = index;
        } else {
            min_index = index + 1;
        }
    }

    cursor->cell_num = min_index;
    return cursor;
}

// leaf_node_insert 定義於 btree.c
// 輔助函式 serialize_row 假設已在 util.c 中實現
void leaf_node_insert(Cursor *cursor, uint32_t key, Row *value) {
    void *node = get_page(cursor->table->pager, cursor->page_num);
    uint32_t num_cells = *leaf_node_num_cells(node);

    if (num_cells >= LEAF_NODE_MAX_CELLS) {
        // 如果節點滿了，需要實現分裂 (Splitting)
        // 這裡暫時視為錯誤並退出
        printf("Error: Node is full. Need to implement splitting node.\n");
        exit(EXIT_FAILURE); 
    }

    if (cursor->cell_num < num_cells) {
        // 插入在中間：需要移動現有單元格，為新的單元格騰出空間
        for (uint32_t i = num_cells; i > cursor->cell_num; i--) {
            // 將 Cell i-1 的數據複製到 Cell i 的位置
            memcpy(leaf_node_cell(node, i), 
                   leaf_node_cell(node, i - 1), 
                   LEAF_NODE_CELL_SIZE);
        }
    }

    // 增加單元格計數
    *leaf_node_num_cells(node) += 1;

    // 將 Key 和 Value 寫入騰出的空間
    *leaf_node_key(node, cursor->cell_num) = key;
    serialize_row(value, leaf_node_value(node, cursor->cell_num));
}
