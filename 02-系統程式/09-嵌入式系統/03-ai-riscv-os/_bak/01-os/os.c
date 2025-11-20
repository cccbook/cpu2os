/* os.c */
#include <stdint.h>
#include <string.h>

/* --- 硬體定義 (QEMU 'virt' machine) --- */
#define CLINT_BASE 0x02000000
#define CLINT_MTIMECMP (CLINT_BASE + 0x4000)
#define CLINT_MTIME    (CLINT_BASE + 0xBFF8)
// 計時器中斷間隔 (例如：10 毫秒, 假設時脈 10MHz)
#define TIMER_INTERVAL 100000 

/* --- mstatus 暫存器位元 --- */
#define MSTATUS_MIE (1 << 3)  /* 全域中斷啟用 */
#define MSTATUS_MPIE (1 << 7) /* 先前的中斷啟用 */

/* --- mie 暫存器位元 --- */
#define MIE_MTIE (1 << 7)     /* M-Mode 計時器中斷啟用 */

/* --- Context 結構 (必須與 trap.S 匹配!) --- */
typedef struct {
    uint32_t ra, sp, gp, tp;
    uint32_t t0, t1, t2;
    uint32_t s0, s1;
    uint32_t a0, a1, a2, a3, a4, a5, a6, a7;
    uint32_t s2, s3, s4, s5, s6, s7, s8, s9, s10, s11;
    uint32_t t3, t4, t5, t6;
    uint32_t mepc;
    uint32_t mstatus;
} Context; // 總共 33 words, 132 bytes

/* --- 任務管理 --- */
#define MAX_TASKS 2
#define STACK_SIZE 256 // 每個任務 256 words (1KB)

// 靜態分配任務上下文和堆疊
Context task_contexts[MAX_TASKS];
uint32_t task_stacks[MAX_TASKS][STACK_SIZE];

// 指向目前和下一個任務的上下文
Context* g_current_task_ctx_ptr;
Context* g_next_task_ctx_ptr;
volatile int current_task_idx = 0;

/* --- 從 trap.S 匯入的函式 --- */
extern void launch_first_task(void);

/* --- 內嵌組合語言 (CSR 讀寫) --- */
static inline uint32_t read_mstatus() {
    uint32_t val;
    asm volatile("csrr %0, mstatus" : "=r"(val));
    return val;
}
static inline void write_mie(uint32_t val) {
    asm volatile("csrw mie, %0" : : "r"(val));
}
static inline void write_mstatus(uint32_t val) {
    asm volatile("csrw mstatus, %0" : : "r"(val));
}

/* --- 任務主體 --- */
// 簡易的 UART 輸出 (QEMU 'virt' 支援)
#define UART0 0x10000000
void uart_puts(char *s) {
    while (*s) {
        *(volatile char *)UART0 = *s++;
    }
}

void task_main_0(void) {
    uart_puts("Task 0: Hello!\n");
    while (1) {
        for (volatile int i = 0; i < 1000000; i++);
        uart_puts("A");
    }
}

void task_main_1(void) {
    uart_puts("Task 1: World!\n");
    while (1) {
        for (volatile int i = 0; i < 1000000; i++);
        uart_puts("B");
    }
}

/* --- 任務初始化 --- */
void task_init(Context* ctx, uint32_t entry_point, uint32_t stack_top) {
    memset(ctx, 0, sizeof(Context));

    // 1. 設置任務堆疊指標 (sp, x2)
    ctx->sp = stack_top;

    // 2. 設置任務起始位址 (mepc)
    ctx->mepc = entry_point;

    // 3. 設置 mstatus
    //    - MPIE 設為 1 (這樣 mret 後能啟用中斷)
    //    - MPP 設為 M-Mode (因為我們都在 M-Mode 執行)
    uint32_t mstatus_val = read_mstatus();
    mstatus_val |= MSTATUS_MPIE;
    // MPP 預設為 M-Mode (0b11), 我們保持不變
    ctx->mstatus = mstatus_val;
}

/* --- 排程器 --- */
void scheduler_init(void) {
    // 初始化任務 0
    task_init(&task_contexts[0],
              (uint32_t)task_main_0,
              (uint32_t)&task_stacks[0][STACK_SIZE]); // 堆疊頂部

    // 初始化任務 1
    task_init(&task_contexts[1],
              (uint32_t)task_main_1,
              (uint32_t)&task_stacks[1][STACK_SIZE]);

    // 設置第一個任務
    current_task_idx = 0;
    g_current_task_ctx_ptr = &task_contexts[0];
    g_next_task_ctx_ptr = &task_contexts[0];
}

// 簡易 Round-Robin 排程器
void scheduler(void) {
    current_task_idx = (current_task_idx + 1) % MAX_TASKS;
    g_next_task_ctx_ptr = &task_contexts[current_task_idx];
}

/* --- 計時器中斷處理 --- */
void timer_tick(void) {
    volatile uint64_t *mtimecmp = (uint64_t*)CLINT_MTIMECMP;
    volatile uint64_t *mtime = (uint64_t*)CLINT_MTIME;

    // 讀取目前時間
    uint64_t now = *mtime;
    // 設置下一次中斷
    uint64_t next_tick = now + TIMER_INTERVAL;
    *mtimecmp = next_tick;
}

/* --- OS 主函式 (由 _start 呼叫) --- */
void os_main(void) {
    uart_puts("OS Starting...\n");

    // 1. 初始化排程器
    scheduler_init();

    // 2. 設置第一次的計時器中斷
    timer_tick();

    // 3. 啟用 M-Mode 計時器中斷 (MTIE)
    write_mie(MIE_MTIE);

    // 4. 全域啟用 M-Mode 中斷 (MIE)
    write_mstatus(read_mstatus() | MSTATUS_MIE);

    uart_puts("Launching first task...\n");

    // 5. 啟動第一個任務 (呼叫組合語言)
    //    ** 這個函式永遠不會返回 **
    launch_first_task();
}
