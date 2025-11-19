// util.h

#ifndef UTIL_H
#define UTIL_H

#include "sq0.h" // 引用 Row 結構的定義

// 為了讓其他模組知道 Row 在磁碟上的大小，雖然也可以在 btree.h 中定義，
// 但放在 util.h 中更符合其作為數據操作工具的定位。

// --- 結構體大小和偏移量 (Row Layout Constants) ---

// ID (uint32_t)
#define ID_SIZE         (sizeof(uint32_t))
#define ID_OFFSET       (0)

// USERNAME (char[32])
#define USERNAME_SIZE   (32)
#define USERNAME_OFFSET (ID_OFFSET + ID_SIZE)

// EMAIL (char[255])
#define EMAIL_SIZE      (255)
#define EMAIL_OFFSET    (USERNAME_OFFSET + USERNAME_SIZE)

// 整個 Row 在磁碟上的總大小 (必須與 Row 結構體大小匹配)
#define ROW_SIZE        (ID_SIZE + USERNAME_SIZE + EMAIL_SIZE)


// --- 函式介面 (Function Prototypes) ---

/**
 * @brief 將一個 Row 結構體轉換為位元組序列 (Serialization)。
 * * @param source 指向要序列化的 Row 結構。
 * @param destination 指向儲存序列化結果的記憶體空間 (必須至少有 ROW_SIZE 大小)。
 */
void serialize_row(Row *source, void *destination);

/**
 * @brief 將位元組序列轉換回一個 Row 結構體 (Deserialization)。
 * * @param source 指向儲存序列化資料的記憶體空間 (必須至少有 ROW_SIZE 大小)。
 * @param destination 指向要填充 Row 結構的記憶體。
 */
void deserialize_row(void *source, Row *destination);

#endif // UTIL_H
