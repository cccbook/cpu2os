#ifndef OS_H
#define OS_H

// 定義 RISC-V 架構
#define RISCV

// 定義布林值
#define TRUE 1
#define FALSE 0

// 定義字節數組的最大長度
#define MAX_BYTES 256

// 定義嵌入式作業系統的狀態
typedef enum {
    OS_OK,
    OS_ERROR
} OS_Status;

// 初始化嵌入式作業系統
OS_Status OS_Init(void);

// 啟動嵌入式作業系統
OS_Status OS_Start(void);

// 停止嵌入式作業系統
OS_Status OS_Stop(void);

// 建立一個新的任務
OS_Status OS_CreateTask(void (*task_function)(void), const char *task_name, unsigned int stack_size);

// 切換到下一個任務
void OS_Schedule(void);

// 傳遞控制權到下一個任務
void OS_Yield(void);

// 睡眠指定時間
void OS_Sleep(unsigned int milliseconds);

#endif // OS_H
