// os.c

#include "os.h"

// 定義作業系統的全局變數
static int os_initialized = FALSE;

// 任務結構
typedef struct {
    void (*task_function)(void);  // 任務函數指針
    const char *task_name;        // 任務名稱
    unsigned int stack_size;      // 任務堆棧大小
    unsigned int sleep_ticks;     // 任務睡眠時間
} OS_Task;

// 定義最大任務數量
#define MAX_TASKS 10

// 保存任務的數組
static OS_Task task_list[MAX_TASKS];

// 保存當前任務的索引
static int current_task_index = 0;

// 當前正在運行的任務索引
static int running_task_index = -1;

// 初始化嵌入式作業系統
OS_Status OS_Init(void) {
    // 在這裡執行初始化工作，例如配置硬體資源、初始化任務調度器等

    // 初始化任務列表
    for (int i = 0; i < MAX_TASKS; ++i) {
        task_list[i].task_function = NULL;
        task_list[i].task_name = NULL;
        task_list[i].stack_size = 0;
        task_list[i].sleep_ticks = 0;
    }

    os_initialized = TRUE;

    return OS_OK;
}

// 創建一個新的任務
OS_Status OS_CreateTask(void (*task_function)(void), const char *task_name, unsigned int stack_size) {
    if (!os_initialized) {
        // 如果作業系統未初始化，返回錯誤
        return OS_ERROR;
    }

    if (current_task_index >= MAX_TASKS) {
        // 如果已達到最大任務數量，返回錯誤
        return OS_ERROR;
    }

    // 將任務參數保存到任務列表中
    task_list[current_task_index].task_function = task_function;
    task_list[current_task_index].task_name = task_name;
    task_list[current_task_index].stack_size = stack_size;
    task_list[current_task_index].sleep_ticks = 0;

    // 增加當前任務索引
    current_task_index++;

    return OS_OK;
}

// 切換到下一個任務
void OS_Schedule(void) {
    if (running_task_index != -1) {
        // 保存當前任務的狀態
        // 可以保存暫存器內容、堆棧指標等
        // 這取決於你的具體硬體平台和任務切換策略
        // ...
    }

    // 選擇下一個任務
    running_task_index = (running_task_index + 1) % current_task_index;

    // 恢復下一個任務的狀態
    // 這可能包括恢復暫存器內容、切換堆棧指標等
    // 這取決於你的具體硬體平台和任務切換策略
    // ...

    // 執行下一個任務
    task_list[running_task_index].task_function();
}

// 傳遞控制權到下一個任務
void OS_Yield(void) {
    // 調用任務調度器，切換到下一個任務
    OS_Schedule();
}

// 睡眠指定時間
void OS_Sleep(unsigned int milliseconds) {
    if (running_task_index != -1) {
        // 設置當前任務的睡眠時間
        task_list[running_task_index].sleep_ticks = milliseconds;

        // 調用任務調度器，切換到下一個任務
        OS_Schedule();
    }
}
