#include "lsm.h"

// 初始化 LSM Tree
void lsm_init(LSMT *tree, size_t capacity) {
    tree->data = (LSMNode **)malloc(capacity * sizeof(LSMNode *));
    if (tree->data == NULL) {
        // 處理內存分配失敗的情況
        perror("LSM Tree initialization failed");
        exit(EXIT_FAILURE);
    }

    tree->size = 0;
    tree->capacity = capacity;
}

// 釋放 LSM Tree 佔用的資源
void lsm_destroy(LSMT *tree) {
    // 釋放每個節點的資源
    for (size_t i = 0; i < tree->size; ++i) {
        free(tree->data[i]->value);
        free(tree->data[i]);
    }

    // 釋放數據陣列本身
    free(tree->data);

    // 重置 tree 結構
    tree->data = NULL;
    tree->size = 0;
    tree->capacity = 0;
}

// 插入鍵值對到 LSM Tree
void lsm_insert(LSMT *tree, uint64_t key, const char *value) {
    // 創建新的節點
    LSMNode *newNode = (LSMNode *)malloc(sizeof(LSMNode));
    if (newNode == NULL) {
        // 處理內存分配失敗的情況
        perror("LSM Node creation failed");
        exit(EXIT_FAILURE);
    }

    // 複製值的內容
    newNode->value = strdup(value);
    if (newNode->value == NULL) {
        // 處理內存分配失敗的情況
        perror("LSM Node value copy failed");
        free(newNode);
        exit(EXIT_FAILURE);
    }

    newNode->key = key;

    // 將新節點插入到數據陣列
    tree->data[tree->size++] = newNode;

    // 如果數據陣列已滿，執行合併操作（根據實際情況擴展）
    if (tree->size == tree->capacity) {
        // 處理數據合併的操作（根據實際情況擴展）
        printf("Performing merge operation...\n");
    }
}

// 查詢指定鍵的值
const char *lsm_query(LSMT *tree, uint64_t key) {
    // 遍歷數據陣列尋找指定鍵
    for (size_t i = 0; i < tree->size; ++i) {
        if (tree->data[i]->key == key) {
            // 返回找到的值
            return tree->data[i]->value;
        }
    }

    // 如果找不到指定鍵，返回空指針
    return NULL;
}

// 刪除指定鍵的鍵值對
void lsm_delete(LSMT *tree, uint64_t key) {
    // 尋找指定鍵的位置
    size_t i;
    for (i = 0; i < tree->size; ++i) {
        if (tree->data[i]->key == key) {
            // 釋放鍵值對的資源
            free(tree->data[i]->value);
            free(tree->data[i]);

            // 將數據陣列中的後續元素向前移動
            for (size_t j = i + 1; j < tree->size; ++j) {
                tree->data[j - 1] = tree->data[j];
            }

            // 減少數據陣列的大小
            tree->size--;

            // 可根據需要進行進一步的合併操作
            // ...

            return;
        }
    }

    // 如果找不到指定鍵，不執行任何操作
}
