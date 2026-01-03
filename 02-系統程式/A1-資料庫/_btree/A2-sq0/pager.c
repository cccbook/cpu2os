// pager.c
#include "pager.h"
#include "sq0.h"
#include <fcntl.h>   // For open()
#include <unistd.h>  // For close(), lseek(), read(), write()
#include <stdlib.h>  // For malloc(), free()
#include <stdio.h>   // For standard I/O and debugging
#include <errno.h>   // For error checking
#include <string.h> // For memset()

// --- Pager 結構與函式實作 ---

// 檢查 I/O 錯誤的通用函式
void check_io_error(int result, const char *action) {
    if (result == -1) {
        perror(action);
        // 在一個真正的資料庫中，這裡會有更優雅的錯誤處理和日誌記錄
        exit(EXIT_FAILURE); 
    }
}

/**
 * @brief 打開資料庫檔案並初始化 Pager 結構。
 * * @param filename 資料庫檔案名稱。
 * @return Pager* 初始化後的 Pager 指標。
 */
Pager *pager_open(const char *filename) {
    // 使用 O_RDWR | O_CREAT 旗標：讀寫模式，如果檔案不存在則創建
    int fd = open(filename, 
                  O_RDWR | O_CREAT, 
                  S_IRUSR | S_IWUSR // 檔案權限：User Read/Write
                 );

    if (fd == -1) {
        printf("Unable to open file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    // 取得檔案長度
    off_t file_length = lseek(fd, 0, SEEK_END);

    Pager *pager = (Pager *)malloc(sizeof(Pager));
    pager->file_descriptor = fd;
    pager->file_length = file_length;
    pager->num_pages = file_length / PAGE_SIZE;

    // 如果檔案長度不是 PAGE_SIZE 的整數倍，則表示檔案可能損壞或不完整
    if (file_length % PAGE_SIZE != 0) {
        printf("Db file is not a whole number of pages. Corrupt file.\n");
        exit(EXIT_FAILURE);
    }

    // 將所有 page 陣列初始化為 NULL
    for (uint32_t i = 0; i < TABLE_MAX_PAGES; i++) {
        pager->pages[i] = NULL;
    }

    return pager;
}

/**
 * @brief 從 Pager 獲取特定頁面編號的資料。
 * 如果頁面已在快取中，直接返回；否則從磁碟讀取。
 * * @param pager Pager 結構指標。
 * @param page_num 頁面編號 (從 0 開始)。
 * @return void* 指向該頁面在記憶體中的資料。
 */
void *get_page(Pager *pager, uint32_t page_num) {
    if (page_num >= TABLE_MAX_PAGES) {
        printf("Tried to fetch page number out of bounds: %u\n", page_num);
        exit(EXIT_FAILURE);
    }

    // 1. 快取命中 (Cache hit)：頁面已在記憶體中
    if (pager->pages[page_num] != NULL) {
        return pager->pages[page_num];
    }

    // 2. 快取未命中 (Cache miss)：需要從磁碟讀取

    // 分配記憶體給新頁面
    void *page = malloc(PAGE_SIZE);
    if (page == NULL) {
        printf("Error: Failed to allocate memory for a new page.\n");
        exit(EXIT_FAILURE);
    }

    uint32_t num_pages = pager->file_length / PAGE_SIZE;

    if (page_num <= num_pages) {
        // 頁面已存在於檔案中，從磁碟讀取
        off_t offset = page_num * PAGE_SIZE;
        int result = lseek(pager->file_descriptor, offset, SEEK_SET);
        check_io_error(result, "Error seeking in db file (read)");

        // 讀取 PAGE_SIZE 位元組到分配的記憶體中
        ssize_t bytes_read = read(pager->file_descriptor, page, PAGE_SIZE);
        if (bytes_read == -1) {
            check_io_error(-1, "Error reading file");
        }
        if ((uint32_t)bytes_read < PAGE_SIZE) {
            // 讀取的位元組數少於期望值，可能為檔案結尾或錯誤
            if (page_num == num_pages) {
                // 如果是最後一個頁面且檔案剛好在結尾，這裡可以接受
                // 但在 PAGE_SIZE 完整度檢查後，這通常不應該發生
            } else {
                printf("Error: Short read from file.\n");
                exit(EXIT_FAILURE);
            }
        }
    } else if (page_num > pager->num_pages) {
         // 這是新的頁面，在檔案末尾，將其歸零
         // 完整的 B-Tree 實作會在這裡處理頁面分配邏輯
         // 暫時將其初始化為 0
         memset(page, 0, PAGE_SIZE);
         pager->num_pages = page_num + 1;
    }


    // 將頁面存入快取
    pager->pages[page_num] = page;
    return page;
}

/**
 * @brief 將指定的頁面資料寫回磁碟。
 * * @param pager Pager 結構指標。
 * @param page_num 要寫回的頁面編號。
 */
void pager_flush(Pager *pager, uint32_t page_num) {
    if (page_num >= TABLE_MAX_PAGES) {
        printf("Tried to flush page number out of bounds: %u\n", page_num);
        exit(EXIT_FAILURE);
    }
    
    // 檢查頁面是否存在於快取中
    if (pager->pages[page_num] == NULL) {
        printf("Tried to flush a null page.\n");
        return;
    }

    // 計算在檔案中的偏移量
    off_t offset = page_num * PAGE_SIZE;
    
    // 移動檔案指標
    int result = lseek(pager->file_descriptor, offset, SEEK_SET);
    check_io_error(result, "Error seeking in db file (flush)");
    
    // 寫入資料
    ssize_t bytes_written = write(pager->file_descriptor, pager->pages[page_num], PAGE_SIZE);
    check_io_error((int)bytes_written, "Error writing file");
}

/**
 * @brief 關閉 Pager，釋放資源。
 * * @param pager Pager 結構指標。
 */
void pager_close(Pager *pager) {
    // 完整的關閉邏輯在 btree.c 的 db_close_table 中處理，
    // 以確保所有髒頁面在關閉檔案前都被 Flush
    
    int result = close(pager->file_descriptor);
    check_io_error(result, "Error closing db file");
    
    free(pager);
}
