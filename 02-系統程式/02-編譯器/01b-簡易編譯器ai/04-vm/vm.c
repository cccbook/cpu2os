// ==========================================================
// file: vm.c
// ==========================================================
#include "ir.h"

// --- VM 資料結構 ---
typedef struct
{
    char name[100];
    int value;
} Variable;

typedef struct
{
    Variable locals[MAX_LOCALS];
    int local_count;
    int return_address;
} StackFrame;

typedef struct
{
    char name[100];
    char params[MAX_ARGS][100];
    int param_count;
    int address;
} FunctionInfo;

typedef struct
{
    IR_Instruction code[MAX_CODE_SIZE];
    int code_size;
    StackFrame call_stack[MAX_CALL_STACK];
    int stack_pointer;
    int ip;
    int retval;
    int running;
    FunctionInfo functions[MAX_FUNCTIONS];
    int function_count;
    int labels[MAX_LABELS];
    char label_names[MAX_LABELS][20];
    int label_count;
    int arg_buffer[MAX_ARGS];
    int arg_count;
} VM;

// --- 載入器 (Loader) ---

// 將字串 op code 轉換回 enum
OpCode string_to_opcode(const char *s)
{
    if (strcmp(s, "OP_ADD") == 0)
        return OP_ADD;
    if (strcmp(s, "OP_SUB") == 0)
        return OP_SUB;
    if (strcmp(s, "OP_MUL") == 0)
        return OP_MUL;
    if (strcmp(s, "OP_DIV") == 0)
        return OP_DIV;
    if (strcmp(s, "OP_EQ") == 0)
        return OP_EQ;
    if (strcmp(s, "OP_NE") == 0)
        return OP_NE;
    if (strcmp(s, "OP_LT") == 0)
        return OP_LT;
    if (strcmp(s, "OP_GT") == 0)
        return OP_GT;
    if (strcmp(s, "OP_LOAD_CONST") == 0)
        return OP_LOAD_CONST;
    if (strcmp(s, "OP_LOAD_VAR") == 0)
        return OP_LOAD_VAR;
    if (strcmp(s, "OP_STORE_VAR") == 0)
        return OP_STORE_VAR;
    if (strcmp(s, "OP_GOTO") == 0)
        return OP_GOTO;
    if (strcmp(s, "OP_IF_FALSE_GOTO") == 0)
        return OP_IF_FALSE_GOTO;
    if (strcmp(s, "OP_LABEL") == 0)
        return OP_LABEL;
    if (strcmp(s, "OP_FUNC_BEGIN") == 0)
        return OP_FUNC_BEGIN;
    if (strcmp(s, "OP_FUNC_END") == 0)
        return OP_FUNC_END;
    if (strcmp(s, "OP_CALL") == 0)
        return OP_CALL;
    if (strcmp(s, "OP_ARG") == 0)
        return OP_ARG;
    if (strcmp(s, "OP_RETURN") == 0)
        return OP_RETURN;
    if (strcmp(s, "OP_GET_RETVAL") == 0)
        return OP_GET_RETVAL;
    return OP_UNKNOWN;
}

// 從檔案載入 IR，填充 VM 的 code 和 functions
void vm_load_ir(VM *vm, const char *filename)
{
    FILE *f = fopen(filename, "r");
    if (!f)
    {
        perror("無法開啟 IR 檔案");
        exit(1);
    }

    char line[256];
    int reading_code = 0;
    while (fgets(line, sizeof(line), f))
    {
        line[strcspn(line, "\n")] = 0;
        if (strlen(line) == 0)
            continue;

        char part1[100] = {0}, part2[100] = {0}, part3[100] = {0}, part4[100] = {0};
        sscanf(line, "%s %s %s %s", part1, part2, part3, part4);

        if (strcmp(part1, "CODE_START") == 0)
        {
            reading_code = 1;
            continue;
        }

        if (!reading_code)
        { // 讀取元資訊
            if (strcmp(part1, "FUNC_INFO") == 0)
            {
                if (vm->function_count >= MAX_FUNCTIONS)
                {
                    fprintf(stderr, "載入錯誤: 函數過多\n");
                    exit(1);
                }
                FunctionInfo *info = &vm->functions[vm->function_count++];
                strcpy(info->name, part2);
                info->param_count = atoi(part3);
                const char *current_pos = strstr(line, part3) + strlen(part3);
                for (int i = 0; i < info->param_count; i++)
                {
                    int read = sscanf(current_pos, " %s", info->params[i]);
                    if (read != 1)
                    {
                        fprintf(stderr, "載入錯誤: 解析函數參數失敗\n");
                        exit(1);
                    }
                    current_pos = strstr(current_pos, info->params[i]) + strlen(info->params[i]);
                }
            }
        }
        else
        { // 讀取指令
            if (vm->code_size >= MAX_CODE_SIZE)
            {
                fprintf(stderr, "載入錯誤: 程式碼過長\n");
                exit(1);
            }
            IR_Instruction *instr = &vm->code[vm->code_size++];
            instr->opcode = string_to_opcode(part1);
            if (instr->opcode == OP_UNKNOWN)
            {
                fprintf(stderr, "載入錯誤: 未知的 opcode '%s'\n", part1);
                exit(1);
            }

            strcpy(instr->result, strcmp(part2, "_") == 0 ? "" : part2);
            strcpy(instr->arg1, strcmp(part3, "_") == 0 ? "" : part3);
            strcpy(instr->arg2, strcmp(part4, "_") == 0 ? "" : part4);
        }
    }
    fclose(f);
}

// --- VM 執行引擎 ---

// 在目前堆疊幀中尋找變數，找不到則報錯
int get_var(VM *vm, const char *name)
{
    StackFrame *current_frame = &vm->call_stack[vm->stack_pointer];
    for (int i = current_frame->local_count - 1; i >= 0; i--)
    {
        if (strcmp(current_frame->locals[i].name, name) == 0)
        {
            return current_frame->locals[i].value;
        }
    }
    fprintf(stderr, "VM 執行錯誤: 在目前作用域找不到變數 '%s'\n", name);
    vm->running = 0;
    return 0; // Return dummy value
}

// 在目前堆疊幀中設定變數，若不存在則建立
void set_var(VM *vm, const char *name, int value)
{
    StackFrame *current_frame = &vm->call_stack[vm->stack_pointer];
    for (int i = 0; i < current_frame->local_count; i++)
    {
        if (strcmp(current_frame->locals[i].name, name) == 0)
        {
            current_frame->locals[i].value = value;
            return;
        }
    }
    if (current_frame->local_count >= MAX_LOCALS)
    {
        fprintf(stderr, "VM 執行錯誤: 區域變數過多\n");
        vm->running = 0;
        return;
    }
    strcpy(current_frame->locals[current_frame->local_count].name, name);
    current_frame->locals[current_frame->local_count].value = value;
    current_frame->local_count++;
}

int find_label_addr(VM *vm, const char *name)
{
    for (int i = 0; i < vm->label_count; i++)
    {
        if (strcmp(vm->label_names[i], name) == 0)
            return vm->labels[i];
    }
    return -1;
}

FunctionInfo *find_function_info(VM *vm, const char *name)
{
    for (int i = 0; i < vm->function_count; i++)
    {
        if (strcmp(vm->functions[i].name, name) == 0)
            return &vm->functions[i];
    }
    return NULL;
}

void vm_init(VM *vm)
{
    vm->ip = 0;
    vm->stack_pointer = -1;
    vm->running = 1;
    vm->label_count = 0;

    // Pre-scan to map labels and function addresses
    for (int i = 0; i < vm->code_size; i++)
    {
        if (vm->code[i].opcode == OP_LABEL)
        {
            if (vm->label_count >= MAX_LABELS)
            {
                fprintf(stderr, "VM 初始化錯誤: 標籤過多\n");
                exit(1);
            }
            strcpy(vm->label_names[vm->label_count], vm->code[i].result);
            vm->labels[vm->label_count] = i;
            vm->label_count++;
        }
        else if (vm->code[i].opcode == OP_FUNC_BEGIN)
        {
            FunctionInfo *info = find_function_info(vm, vm->code[i].result);
            if (info)
            {
                info->address = i;
            }
            else
            {
                fprintf(stderr, "VM 初始化錯誤: IR 包含未在元資訊中定義的函數 '%s'\n", vm->code[i].result);
                exit(1);
            }
        }
    }
}

void vm_run(VM *vm)
{
    FunctionInfo *main_func = find_function_info(vm, "main");
    if (!main_func)
    {
        fprintf(stderr, "VM 錯誤: 找不到 main 函數\n");
        return;
    }
    if (main_func->address < 0)
    {
        fprintf(stderr, "VM 錯誤: main 函數地址未初始化\n");
        return;
    }

    vm->stack_pointer = 0;
    vm->call_stack[0].local_count = 0;
    vm->call_stack[0].return_address = -1;
    vm->ip = main_func->address;

    while (vm->running && vm->ip < vm->code_size)
    {
        IR_Instruction instr = vm->code[vm->ip];
        vm->ip++;

        switch (instr.opcode)
        {
        case OP_LOAD_CONST:
            set_var(vm, instr.result, atoi(instr.arg1));
            break;
        case OP_LOAD_VAR:
            set_var(vm, instr.result, get_var(vm, instr.arg1));
            break;
        case OP_STORE_VAR:
            set_var(vm, instr.result, get_var(vm, instr.arg1));
            break;
        case OP_ADD:
            set_var(vm, instr.result, get_var(vm, instr.arg1) + get_var(vm, instr.arg2));
            break;
        case OP_SUB:
            set_var(vm, instr.result, get_var(vm, instr.arg1) - get_var(vm, instr.arg2));
            break;
        case OP_MUL:
            set_var(vm, instr.result, get_var(vm, instr.arg1) * get_var(vm, instr.arg2));
            break;
        case OP_DIV:
            set_var(vm, instr.result, get_var(vm, instr.arg1) / get_var(vm, instr.arg2));
            break;
        case OP_EQ:
            set_var(vm, instr.result, get_var(vm, instr.arg1) == get_var(vm, instr.arg2));
            break;

        case OP_LABEL:
        case OP_FUNC_BEGIN:
        case OP_FUNC_END:
            break;

        case OP_ARG:
            if (vm->arg_count >= MAX_ARGS)
            {
                fprintf(stderr, "VM 錯誤: 參數過多\n");
                vm->running = 0;
                break;
            }
            vm->arg_buffer[vm->arg_count++] = get_var(vm, instr.arg1);
            break;
        case OP_CALL:
        {
            FunctionInfo *func_info = find_function_info(vm, instr.arg1);
            if (!func_info)
            {
                fprintf(stderr, "VM 錯誤: 呼叫未定義的函數 %s\n", instr.arg1);
                vm->running = 0;
                break;
            }
            if (vm->stack_pointer + 1 >= MAX_CALL_STACK)
            {
                fprintf(stderr, "VM 錯誤: 呼叫堆疊溢位\n");
                vm->running = 0;
                break;
            }

            vm->stack_pointer++;
            StackFrame *new_frame = &vm->call_stack[vm->stack_pointer];
            new_frame->return_address = vm->ip;
            new_frame->local_count = 0;

            for (int i = 0; i < func_info->param_count; i++)
            {
                set_var(vm, func_info->params[i], vm->arg_buffer[i]);
            }
            vm->arg_count = 0;
            vm->ip = func_info->address;
            break;
        }
        case OP_RETURN:
        {
            if (strlen(instr.arg1) > 0)
                vm->retval = get_var(vm, instr.arg1);

            int return_addr = vm->call_stack[vm->stack_pointer].return_address;
            vm->stack_pointer--;

            if (vm->stack_pointer < -1 || return_addr == -1)
            {
                vm->running = 0; // End of program
            }
            else
            {
                vm->ip = return_addr;
            }
            break;
        }
        case OP_GET_RETVAL:
            set_var(vm, instr.result, vm->retval);
            break;

        default:
            fprintf(stderr, "VM 錯誤: 在位址 %d 遇到未知的指令 %d\n", vm->ip - 1, instr.opcode);
            vm->running = 0;
            break;
        }
    }
}

// --- 主程式 ---
int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        fprintf(stderr, "用法: %s <ir_file>\n", argv[0]);
        return 1;
    }

    // 1. 初始化 VM 並從檔案載入 bytecode
    VM vm = {0}; // Initialize all fields to zero
    printf("--- 虛擬機執行階段 ---\n");
    printf("從 '%s' 載入中間碼...\n", argv[1]);
    vm_load_ir(&vm, argv[1]);

    // 2. 初始化 VM 執行環境 (掃描標籤等)
    vm_init(&vm);

    // 3. 執行
    printf("執行 bytecode...\n\n");
    vm_run(&vm);

    // 4. 輸出結果
    if (vm.running == 0 && vm.stack_pointer != -1)
    {
        printf("\n--- 執行因錯誤而中止 ---\n");
    }
    else
    {
        printf("\n--- 執行結果 ---\n");
        printf("main 函數回傳值: %d\n", vm.retval);
    }

    return 0;
}