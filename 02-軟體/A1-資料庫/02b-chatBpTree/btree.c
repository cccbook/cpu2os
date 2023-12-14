#include "btree.h"

void splitChild(BTreeNode *parent, int index);
void insertNonFull(BTreeNode *node, int key);
void deleteKeyHelper(BTreeNode *node, int key);
void mergeChildren(BTreeNode *node, int index);
BTreeNode *getSibling(BTreeNode *node, int index);
void removeFromLeaf(BTreeNode *node, int index);
void removeFromInternal(BTreeNode *node, int index);
int getPredecessor(BTreeNode *node);
int getSuccessor(BTreeNode *node);
void borrowFromRightSibling(BTreeNode *node, int index);
void borrowFromLeftSibling(BTreeNode *node, int index);
void mergeWithLeftSibling(BTreeNode *node, int index);
void mergeWithRightSibling(BTreeNode *node, int index);
int findKeyIndex(BTreeNode *node, int key);

BTree *createBTree() {
    BTree *tree = (BTree *)malloc(sizeof(BTree));
    if (!tree) {
        perror("Memory allocation failed");
        exit(EXIT_FAILURE);
    }

    tree->root = NULL;

    return tree;
}

BTreeNode *createNode() {
    BTreeNode *node = (BTreeNode *)malloc(sizeof(BTreeNode));
    if (!node) {
        perror("Memory allocation failed");
        exit(EXIT_FAILURE);
    }

    node->leaf = 1; // 新建節點默認為葉子節點
    node->num_keys = 0;

    for (int i = 0; i < 2 * DEGREE; ++i) {
        node->keys[i] = 0;
        node->children[i] = NULL;
    }

    return node;
}

void insert(BTree *tree, int key) {
    BTreeNode *root = tree->root;

    // 如果樹為空，則創建一個新的根節點
    if (!root) {
        tree->root = createNode();
        tree->root->keys[0] = key;
        tree->root->num_keys = 1;
    } else {
        // 如果根節點已滿，則分裂根節點
        if (root->num_keys == 2 * DEGREE - 1) {
            BTreeNode *newRoot = createNode();
            tree->root = newRoot;
            newRoot->leaf = 0;
            newRoot->children[0] = root;

            // 分裂根節點
            // 將原根節點分裂成兩個子節點，新的根節點包含兩者的中間鍵
            splitChild(newRoot, 0);
            // 插入鍵到分裂後的子節點
            insertNonFull(newRoot, key);
        } else {
            // 如果根節點不滿，則直接插入
            insertNonFull(root, key);
        }
    }
}

// 具體的分裂子節點的函數
void splitChild(BTreeNode *parent, int index) {
    BTreeNode *child = parent->children[index];
    BTreeNode *newChild = createNode();
    
    newChild->leaf = child->leaf;
    newChild->num_keys = DEGREE - 1;

    // 將 child 中的後半部分複製到 newChild
    for (int i = 0; i < DEGREE - 1; ++i) {
        newChild->keys[i] = child->keys[i + DEGREE];
    }

    // 更新 child 的鍵數量
    child->num_keys = DEGREE - 1;

    // 將 newChild 插入到 parent 中
    for (int i = parent->num_keys; i > index; --i) {
        parent->children[i + 1] = parent->children[i];
    }
    parent->children[index + 1] = newChild;

    // 將 child 中的中間鍵插入到 parent 中
    for (int i = parent->num_keys - 1; i >= index; --i) {
        parent->keys[i + 1] = parent->keys[i];
    }
    parent->keys[index] = child->keys[DEGREE - 1];
    parent->num_keys++;
}

// 在非滿節點中插入鍵的函數
void insertNonFull(BTreeNode *node, int key) {
    int i = node->num_keys - 1;

    // 如果節點是葉子節點，則直接插入鍵
    if (node->leaf) {
        while (i >= 0 && key < node->keys[i]) {
            node->keys[i + 1] = node->keys[i];
            i--;
        }
        node->keys[i + 1] = key;
        node->num_keys++;
    } else {
        // 如果節點是內部節點，找到適當的子節點插入
        while (i >= 0 && key < node->keys[i]) {
            i--;
        }

        i++;

        // 如果子節點已滿，則分裂子節點
        if (node->children[i]->num_keys == 2 * DEGREE - 1) {
            splitChild(node, i);

            // 確定插入到新分裂的子節點還是舊的子節點
            if (key > node->keys[i]) {
                i++;
            }
        }

        // 遞歸插入
        insertNonFull(node->children[i], key);
    }
}

void deleteKey(BTree *tree, int key) {
    if (tree->root == NULL) {
        printf("BTree is empty.\n");
        return;
    }

    // 處理根節點
    if (tree->root->num_keys == 1 && tree->root->leaf == 1) {
        // 如果根節點是葉子節點且只有一個鍵，直接刪除
        if (tree->root->keys[0] == key) {
            free(tree->root);
            tree->root = NULL;
            return;
        } else {
            printf("Key not found in BTree.\n");
            return;
        }
    }

    // 呼叫遞歸刪除的輔助函數
    deleteKeyHelper(tree->root, key);

    // 如果根節點只有一個子節點，則更新根節點
    if (tree->root->num_keys == 0) {
        BTreeNode *newRoot = tree->root->children[0];
        free(tree->root);
        tree->root = newRoot;
    }
}

// 刪除鍵的輔助函數
void deleteKeyHelper(BTreeNode *node, int key) {
    int index = findKeyIndex(node, key);

    // 如果鍵在當前節點
    if (index < node->num_keys && node->keys[index] == key) {
        // 處理葉子節點的情況
        if (node->leaf) {
            removeFromLeaf(node, index);
        } else {
            // 處理內部節點的情況
            removeFromInternal(node, index);
        }
    } else {
        // 如果鍵不在當前節點，找到包含鍵的子節點
        BTreeNode *child = node->children[index];
        if (child->num_keys >= DEGREE) {
            // 如果子節點中的鍵數量足夠，遞歸刪除
            deleteKeyHelper(child, key);
        } else {
            // 如果子節點中的鍵數量不足，進行合併或重新分配
            BTreeNode *sibling = getSibling(node, index);

            if (sibling != NULL && sibling->num_keys >= DEGREE) {
                // 如果有右鄰居且右鄰居有足夠的鍵，借一個鍵
                borrowFromRightSibling(node, index);
            } else if (sibling != NULL && sibling->num_keys == DEGREE - 1) {
                // 如果右鄰居的鍵數量不足，與右鄰居合併
                mergeWithRightSibling(node, index);
            } else if (index > 0) {
                // 如果沒有右鄰居，但有左鄰居，借一個鍵
                borrowFromLeftSibling(node, index);
            } else {
                // 如果左鄰居的鍵數量不足，與左鄰居合併
                mergeWithLeftSibling(node, index);
            }

            // 遞歸刪除
            deleteKeyHelper(node->children[index], key);
        }
    }
}

// 在葉子節點中刪除鍵
void removeFromLeaf(BTreeNode *node, int index) {
    for (int i = index + 1; i < node->num_keys; ++i) {
        node->keys[i - 1] = node->keys[i];
    }

    node->num_keys--;
}

// 在內部節點中刪除鍵
void removeFromInternal(BTreeNode *node, int index) {
    int key = node->keys[index];

    // 如果左子節點的鍵數量大於等於 DEGREE，找到前驅鍵
    if (node->children[index]->num_keys >= DEGREE) {
        int predecessor = getPredecessor(node->children[index]);
        node->keys[index] = predecessor;
        deleteKeyHelper(node->children[index], predecessor);
    } else if (node->children[index + 1]->num_keys >= DEGREE) {
        // 如果右子節點的鍵數量大於等於 DEGREE，找到後繼鍵
        int successor = getSuccessor(node->children[index + 1]);
        node->keys[index] = successor;
        deleteKeyHelper(node->children[index + 1], successor);
    } else {
        // 如果左右子節點的鍵數量都不足，合併左右子節點
        mergeChildren(node, index);
        deleteKeyHelper(node->children[index], key);
    }
}

// 從左兄弟節點中借一個鍵
void borrowFromLeftSibling(BTreeNode *node, int index) {
    BTreeNode *child = node->children[index];
    BTreeNode *leftSibling = node->children[index - 1];

    // 將父節點的鍵和左兄弟的最後一個鍵下移到 child
    for (int i = child->num_keys; i > 0; --i) {
        child->keys[i] = child->keys[i - 1];
    }
    child->keys[0] = node->keys[index - 1];
    node->keys[index - 1] = leftSibling->keys[leftSibling->num_keys - 1];

    if (!child->leaf) {
        // 如果 child 不是葉子節點，還需要處理子節點的指針
        for (int i = child->num_keys + 1; i > 0; --i) {
            child->children[i] = child->children[i - 1];
        }
        child->children[0] = leftSibling->children[leftSibling->num_keys];
    }

    child->num_keys++;
    leftSibling->num_keys--;
}

// 從右兄弟節點中借一個鍵
void borrowFromRightSibling(BTreeNode *node, int index) {
    BTreeNode *child = node->children[index];
    BTreeNode *rightSibling = node->children[index + 1];

    // 將父節點的鍵和右兄弟的第一個鍵下移到 child
    child->keys[child->num_keys] = node->keys[index];
    node->keys[index] = rightSibling->keys[0];

    if (!child->leaf) {
        // 如果 child 不是葉子節點，還需要處理子節點的指針
        child->children[child->num_keys + 1] = rightSibling->children[0];
    }

    // 將右兄弟的鍵左移一位
    for (int i = 0; i < rightSibling->num_keys - 1; ++i) {
        rightSibling->keys[i] = rightSibling->keys[i + 1];
    }

    if (!rightSibling->leaf) {
        // 如果右兄弟不是葉子節點，還需要處理子節點的指針
        rightSibling->children[0] = rightSibling->children[1];
    }

    child->num_keys++;
    rightSibling->num_keys--;
}

// 合併左右兄弟節點
void mergeChildren(BTreeNode *node, int index) {
    BTreeNode *child = node->children[index];
    BTreeNode *rightSibling = node->children[index + 1];

    // 將父節點的鍵和右兄弟的所有鍵合併到 child
    child->keys[DEGREE - 1] = node->keys[index];

    for (int i = 0; i < rightSibling->num_keys; ++i) {
        child->keys[DEGREE + i] = rightSibling->keys[i];
    }

    if (!child->leaf) {
        // 如果 child 不是葉子節點，還需要處理子節點的指針
        for (int i = 0; i <= rightSibling->num_keys; ++i) {
            child->children[DEGREE + i] = rightSibling->children[i];
        }
    }

    // 將右兄弟的內容往前移
    for (int i = index + 1; i < node->num_keys; ++i) {
        node->keys[i - 1] = node->keys[i];
        node->children[i] = node->children[i + 1];
    }

    node->num_keys--;
    child->num_keys += rightSibling->num_keys + 1;

    free(rightSibling);
}

// 找到子節點的前驅鍵
int getPredecessor(BTreeNode *node) {
    while (!node->leaf) {
        node = node->children[node->num_keys];
    }
    return node->keys[node->num_keys - 1];
}

// 找到子節點的後繼鍵
int getSuccessor(BTreeNode *node) {
    while (!node->leaf) {
        node = node->children[0];
    }
    return node->keys[0];
}

// 找到包含鍵的子節點的索引
int findKeyIndex(BTreeNode *node, int key) {
    int index = 0;
    while (index < node->num_keys && key > node->keys[index]) {
        index++;
    }
    return index;
}

// 找到左右兄弟節點
BTreeNode *getSibling(BTreeNode *node, int index) {
    if (index > 0) {
        return node->children[index - 1];
    } else if (index < node->num_keys) {
        return node->children[index + 1];
    } else {
        return NULL;
    }
}

// 遞歸釋放節點的函數
void destroyNode(BTreeNode *node) {
    if (node) {
        if (!node->leaf) {
            for (int i = 0; i <= node->num_keys; ++i) {
                destroyNode(node->children[i]);
            }
        }
        free(node);
    }
}

// 遞歸釋放 B+Tree 的函數
void destroyBTree(BTree *tree) {
    if (tree) {
        destroyNode(tree->root);
        free(tree);
    }
}

// 遞歸打印節點的函數
void printNode(BTreeNode *node, int level) {
    if (node) {
        printf("Level %d: ", level);

        for (int i = 0; i < node->num_keys; ++i) {
            printf("%d ", node->keys[i]);
        }
        printf("\n");

        if (!node->leaf) {
            for (int i = 0; i <= node->num_keys; ++i) {
                printNode(node->children[i], level + 1);
            }
        }
    }
}

// 打印 B+Tree 的函數
void printBTree(BTree *tree) {
    if (tree) {
        printNode(tree->root, 0);
    }
}

// ===================================

// 合併節點與右鄰居節點
void mergeWithRightSibling(BTreeNode *node, int index) {
    BTreeNode *child = node->children[index];
    BTreeNode *rightSibling = node->children[index + 1];

    // 將父節點的鍵下移到 child
    child->keys[DEGREE - 1] = node->keys[index];

    // 將右鄰居的所有鍵複製到 child
    for (int i = 0; i < rightSibling->num_keys; ++i) {
        child->keys[DEGREE + i] = rightSibling->keys[i];
    }

    if (!child->leaf) {
        // 如果 child 不是葉子節點，還需要處理子節點的指針
        for (int i = 0; i <= rightSibling->num_keys; ++i) {
            child->children[DEGREE + i] = rightSibling->children[i];
        }
    }

    // 將右鄰居的內容往前移
    for (int i = index + 1; i < node->num_keys; ++i) {
        node->keys[i - 1] = node->keys[i];
        node->children[i] = node->children[i + 1];
    }

    node->num_keys--;
    child->num_keys += rightSibling->num_keys + 1;

    free(rightSibling);
}

// 合併節點與左鄰居節點
void mergeWithLeftSibling(BTreeNode *node, int index) {
    BTreeNode *child = node->children[index];
    BTreeNode *leftSibling = node->children[index - 1];

    // 將父節點的鍵下移到 leftSibling
    leftSibling->keys[leftSibling->num_keys] = node->keys[index - 1];

    // 將 child 的所有鍵複製到 leftSibling
    for (int i = 0; i < child->num_keys; ++i) {
        leftSibling->keys[leftSibling->num_keys + 1 + i] = child->keys[i];
    }

    if (!child->leaf) {
        // 如果 child 不是葉子節點，還需要處理子節點的指針
        for (int i = 0; i <= child->num_keys; ++i) {
            leftSibling->children[leftSibling->num_keys + 1 + i] = child->children[i];
        }
    }

    // 將父節點的相應鍵往前移
    for (int i = index; i < node->num_keys - 1; ++i) {
        node->keys[i - 1] = node->keys[i];
        node->children[i] = node->children[i + 1];
    }

    node->num_keys--;
    leftSibling->num_keys += child->num_keys + 1;

    free(child);
}

