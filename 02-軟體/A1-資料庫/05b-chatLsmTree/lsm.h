#ifndef LSM_H
#define LSM_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// 定義 LSM Tree 节點的結構
typedef struct {
    uint64_t key;
    char *value;
} LSMNode;

// 定義 LSM Tree 結構
typedef struct {
    LSMNode **data;   // LSM Tree 數據陣列
    size_t size;      // 當前數據陣列的大小
    size_t capacity;  // 數據陣列的容量
} LSMT;

// 初始化 LSM Tree
void lsm_init(LSMT *tree, size_t capacity);

// 釋放 LSM Tree 佔用的資源
void lsm_destroy(LSMT *tree);

// 插入鍵值對到 LSM Tree
void lsm_insert(LSMT *tree, uint64_t key, const char *value);

// 查詢指定鍵的值
const char *lsm_query(LSMT *tree, uint64_t key);

// 刪除指定鍵的鍵值對
void lsm_delete(LSMT *tree, uint64_t key);

#endif /* LSM_H */
