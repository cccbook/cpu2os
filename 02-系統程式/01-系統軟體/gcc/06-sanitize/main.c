#include <stdio.h>
#include <stdlib.h>

void cause_error() {
    // 宣告一個只有 10 個元素的整數陣列
    int buffer[10]; 
    int i;
    
    printf("--- 嘗試寫入超出緩衝區邊界的記憶體 ---\n");
    
    // 故意存取第 10 個索引 (第 11 個元素)，這是越界存取
    // 合法的索引是 0 到 9
    i = 10; 
    
    // ASan 會在這裡偵測到 'stack-buffer-overflow'
    buffer[i] = 42; 

    printf("這行程式碼可能不會被執行到（取決於編譯器和系統）\n");
}

int main() {
    printf("程式開始執行\n");
    cause_error();
    printf("程式正常結束 (如果錯誤沒有被捕捉到)\n");
    return 0;
}