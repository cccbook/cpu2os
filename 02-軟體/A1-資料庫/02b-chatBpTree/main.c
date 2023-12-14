#include "btree.h"

int main() {
    BTree *tree = createBTree();

    // 插入一些鍵
    insert(tree, 10);
    insert(tree, 20);
    insert(tree, 5);
    insert(tree, 6);
    insert(tree, 12);
    insert(tree, 30);
    insert(tree, 7);
    insert(tree, 17);

    // 打印 B+Tree
    printf("B+Tree after insertions:\n");
    printBTree(tree);

    // 刪除一些鍵
    deleteKey(tree, 6);
    deleteKey(tree, 30);

    // 打印 B+Tree
    printf("\nB+Tree after deletions:\n");
    printBTree(tree);

    // 釋放 B+Tree 的內存
    destroyBTree(tree);

    return 0;
}
