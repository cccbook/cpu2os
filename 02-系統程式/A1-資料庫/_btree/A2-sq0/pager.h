// pager.h

#ifndef PAGER_H
#define PAGER_H

#include <stdio.h>
#include <stdint.h>
#include "sq0.h" // 引入 PAGE_SIZE

// 頁面快取結構
typedef struct {
    int file_descriptor;
    uint32_t file_length;
    uint32_t num_pages; // 記錄目前資料庫檔案有多少頁
    void *pages[TABLE_MAX_PAGES]; // 儲存頁面資料的陣列
} Pager;

// 函式介面
Pager *pager_open(const char *filename);
void *get_page(Pager *pager, uint32_t page_num);
void pager_flush(Pager *pager, uint32_t page_num); // 將頁面寫回磁碟
void pager_close(Pager *pager);

#endif // PAGER_H