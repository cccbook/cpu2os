#include "lsm.h"

int main() {
    // 初始化 LSM Tree
    LSMT tree;
    lsm_init(&tree, 5);

    // 插入鍵值對
    lsm_insert(&tree, 10, "value1");
    lsm_insert(&tree, 5, "value2");
    lsm_insert(&tree, 20, "value3");

    // 查詢鍵值對
    const char *result1 = lsm_query(&tree, 10);
    const char *result2 = lsm_query(&tree, 5);
    const char *result3 = lsm_query(&tree, 20);
    const char *result4 = lsm_query(&tree, 15);

    // 刪除鍵值對
    lsm_delete(&tree, 5);

    // 再次查詢鍵值對
    const char *result5 = lsm_query(&tree, 5);

    // 釋放 LSM Tree 佔用的資源
    lsm_destroy(&tree);

    // 輸出結果
    printf("Result 1: %s\n", result1);
    printf("Result 2: %s\n", result2);
    printf("Result 3: %s\n", result3);
    printf("Result 4: %s\n", result4);
    printf("Result 5: %s\n", result5);

    return 0;
}
