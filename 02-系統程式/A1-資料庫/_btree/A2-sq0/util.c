// util.c

#include "sq0.h"
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// --- 欄位偏移量 (Field Offsets) ---
// 為了確保資料在儲存時與 C 語言結構體的對齊方式無關，
// 我們需要精確計算每個欄位在位元組序列中的起始位置。

// ID (uint32_t)
const uint32_t ID_SIZE = sizeof(uint32_t);
const uint32_t ID_OFFSET = 0;

// USERNAME (char[32])
const uint32_t USERNAME_SIZE = 32;
const uint32_t USERNAME_OFFSET = ID_OFFSET + ID_SIZE;

// EMAIL (char[255])
const uint32_t EMAIL_SIZE = 255;
const uint32_t EMAIL_OFFSET = USERNAME_OFFSET + USERNAME_SIZE;

// 整個 Row 的總大小
const uint32_t ROW_SIZE = ID_SIZE + USERNAME_SIZE + EMAIL_SIZE;


/**
 * @brief 將一個 Row 結構體轉換為位元組序列 (Serialization)。
 * * @param source 指向要序列化的 Row 結構。
 * @param destination 指向儲存序列化結果的記憶體空間。
 */
void serialize_row(Row *source, void *destination) {
    // 1. 序列化 ID (uint32_t)
    // 注意：這裡假設所有系統都使用相同的位元組順序 (Endianness)。
    // 在一個真正的資料庫中，ID 會以網路位元組順序 (Big Endian) 儲存，以確保跨平台相容性。
    // 在 C 實作中，可以透過 htonl() 函式來處理位元組順序，但這裡先使用 memcpy 簡化。
    memcpy(destination + ID_OFFSET, &(source->id), ID_SIZE);

    // 2. 序列化 USERNAME (char[32])
    // 必須使用 strncpy 確保不會寫入超過 32 位元組，並確保剩餘空間被填零 (以防止垃圾數據)
    // 注意：strncpy 不保證以空字元結尾，所以手動填零是更安全的做法。
    memset(destination + USERNAME_OFFSET, 0, USERNAME_SIZE);
    strncpy(destination + USERNAME_OFFSET, source->username, USERNAME_SIZE - 1);

    // 3. 序列化 EMAIL (char[255])
    memset(destination + EMAIL_OFFSET, 0, EMAIL_SIZE);
    strncpy(destination + EMAIL_OFFSET, source->email, EMAIL_SIZE - 1);
}

/**
 * @brief 將位元組序列轉換回一個 Row 結構體 (Deserialization)。
 * * @param source 指向儲存序列化資料的記憶體空間。
 * @param destination 指向要填充 Row 結構的記憶體。
 */
void deserialize_row(void *source, Row *destination) {
    // 1. 反序列化 ID
    memcpy(&(destination->id), source + ID_OFFSET, ID_SIZE);

    // 2. 反序列化 USERNAME
    // 從位元組序列複製到結構體的陣列中
    memcpy(destination->username, source + USERNAME_OFFSET, USERNAME_SIZE);
    // 確保以空字元結尾，以防萬一 (因為我們在序列化時已保證了，但這是一個安全做法)
    destination->username[USERNAME_SIZE - 1] = '\0'; 

    // 3. 反序列化 EMAIL
    memcpy(destination->email, source + EMAIL_OFFSET, EMAIL_SIZE);
    destination->email[EMAIL_SIZE - 1] = '\0';
}
