#ifndef BTREE_H
#define BTREE_H

#include <stdio.h>
#include <stdlib.h>

// 定義 B+Tree 的度數
#define DEGREE 3

// 定義節點的結構
typedef struct BTreeNode {
    int leaf; // 1 if leaf, 0 if internal
    int num_keys; // 當前節點中的鍵數量
    int keys[2 * DEGREE - 1]; // 鍵的數組
    struct BTreeNode *children[2 * DEGREE]; // 子節點的指針數組
    // 可以加入其他資料成員，視需要而定
} BTreeNode;

// 定義 B+Tree 的結構
typedef struct BTree {
    BTreeNode *root; // 樹的根節點
    // 可以加入其他資料成員，視需要而定
} BTree;

// 函數原型宣告
BTree *createBTree();
BTreeNode *createNode();
void insert(BTree *tree, int key);
void deleteKey(BTree *tree, int key);
void destroyBTree(BTree *tree);
void printBTree(BTree *tree);

#endif // BTREE_H
