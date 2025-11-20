// ==========================================================
// file: ir2asm.c (macOS Professional Format)
// ==========================================================
#include "ir.h"

#define MAX_VARS 50
#define MAX_TEMPS 50
/*
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_CODE_SIZE 1024
#define MAX_FUNCTIONS 50
#define MAX_VARS 50
#define MAX_TEMPS 50

// --- IR 定義 ---
typedef enum
{
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    OP_EQ,
    OP_NE,
    OP_LT,
    OP_GT,
    OP_LOAD_CONST,
    OP_LOAD_VAR,
    OP_STORE_VAR,
    OP_GOTO,
    OP_IF_FALSE_GOTO,
    OP_LABEL,
    OP_FUNC_BEGIN,
    OP_FUNC_END,
    OP_CALL,
    OP_ARG,
    OP_RETURN,
    OP_GET_RETVAL,
    OP_UNKNOWN
} OpCode;
typedef struct
{
    OpCode opcode;
    char result[20];
    char arg1[20];
    char arg2[20];
} IR_Instruction;
*/

typedef struct
{
    char name[100];
    char params[10][100];
    int param_count;
} FunctionInfo;

// --- 狀態管理結構 ---
typedef struct
{
    char name[100];
    int offset;
} VarMap;
typedef struct
{
    char name[20];
    int reg_idx;
} TempMap;

typedef struct
{
    FunctionInfo *info;
    VarMap var_map[MAX_VARS];
    int var_count;
    TempMap temp_map[MAX_TEMPS];
    int temp_count;
    int stack_size;
} FuncContext;

FunctionInfo *find_function_info(FunctionInfo *funcs, int count, const char *name)
{
    for (int i = 0; i < count; i++)
    {
        if (strcmp(funcs[i].name, name) == 0)
            return &funcs[i];
    }
    return NULL;
}

// --- 翻譯器輔助函式 ---
int get_temp_reg(FuncContext *ctx, const char *temp_name)
{
    for (int i = 0; i < ctx->temp_count; i++)
        if (strcmp(ctx->temp_map[i].name, temp_name) == 0)
            return ctx->temp_map[i].reg_idx;
    int reg_idx = 8 + ctx->temp_count;
    strcpy(ctx->temp_map[ctx->temp_count].name, temp_name);
    ctx->temp_map[ctx->temp_count].reg_idx = reg_idx;
    ctx->temp_count++;
    return reg_idx;
}

int get_var_offset(FuncContext *ctx, const char *var_name)
{
    for (int i = 0; i < ctx->var_count; i++)
        if (strcmp(ctx->var_map[i].name, var_name) == 0)
            return ctx->var_map[i].offset;
    // New variable, allocate space on stack
    ctx->stack_size += 4;
    strcpy(ctx->var_map[ctx->var_count].name, var_name);
    ctx->var_map[ctx->var_count].offset = -ctx->stack_size;
    ctx->var_count++;
    return -ctx->stack_size;
}

void translate_function(FILE *out, FunctionInfo *func_info, IR_Instruction *all_code, int start_idx)
{
    FuncContext ctx = {0};
    ctx.info = func_info;
    int current_arg_reg = 0;
    int stack_space_for_locals = 0;

    // --- Pass 1: Scan to calculate stack size for local variables ---
    for (int i = start_idx; all_code[i].opcode != OP_FUNC_END; i++)
    {
        if (all_code[i].opcode == OP_STORE_VAR)
        {
            int is_param = 0;
            for (int p = 0; p < func_info->param_count; p++)
                if (strcmp(all_code[i].result, func_info->params[p]) == 0)
                    is_param = 1;

            int is_mapped = 0;
            for (int v = 0; v < ctx.var_count; v++)
                if (strcmp(all_code[i].result, ctx.var_map[v].name) == 0)
                    is_mapped = 1;

            if (!is_param && !is_mapped)
            {
                strcpy(ctx.var_map[ctx.var_count++].name, all_code[i].result);
                stack_space_for_locals += 4;
            }
        }
    }
    // Total stack space needed (for fp/lr + locals), must be 16-byte aligned
    int total_stack_alloc = (16 + stack_space_for_locals + 15) & -16;

    // Assign offsets to locals
    int current_local_offset = -16 - 4; // Start after saved fp/lr
    for (int i = 0; i < ctx.var_count; i++)
    {
        ctx.var_map[i].offset = current_local_offset;
        current_local_offset -= 4;
    }

    // --- Pass 2: Generate Assembly ---
    fprintf(out, "\t.globl\t_%s\t\t\t\t; -- Begin function %s\n", func_info->name, func_info->name);
    fprintf(out, "\t.p2align\t2\n");
    fprintf(out, "_%s:\t\t\t\t\t; @%s\n", func_info->name, func_info->name);
    fprintf(out, "\t.cfi_startproc\n");

    // Prologue
    fprintf(out, "\tsub\tsp, sp, #%d\n", total_stack_alloc);
    fprintf(out, "\t.cfi_def_cfa_offset %d\n", total_stack_alloc);
    fprintf(out, "\tstp\tx29, x30, [sp, #%d]\n", total_stack_alloc - 16);

    // Store parameters from registers to stack
    for (int i = 0; i < func_info->param_count; i++)
    {
        get_var_offset(&ctx, func_info->params[i]); // This will map the param
        fprintf(out, "\tstr\tw%d, [sp, #%d]\t\t; Store param '%s'\n", i, 12 - (i * 4), func_info->params[i]);
    }

    // Translate instructions
    for (int i = start_idx + 1; all_code[i].opcode != OP_FUNC_END; i++)
    {
        IR_Instruction instr = all_code[i];
        switch (instr.opcode)
        {
        case OP_LOAD_CONST:
            fprintf(out, "\tmov\tw%d, #%s\n", get_temp_reg(&ctx, instr.result), instr.arg1);
            break;
        case OP_LOAD_VAR:
            fprintf(out, "\tldr\tw%d, [sp, #%d]\t\t; Load var '%s'\n", get_temp_reg(&ctx, instr.result), get_var_offset(&ctx, instr.arg1), instr.arg1);
            break;
        case OP_STORE_VAR:
            fprintf(out, "\tstr\tw%d, [sp, #%d]\t\t; Store var '%s'\n", get_temp_reg(&ctx, instr.arg1), get_var_offset(&ctx, instr.result), instr.result);
            break;
        case OP_ADD:
            fprintf(out, "\tadd\tw%d, w%d, w%d\n", get_temp_reg(&ctx, instr.result), get_temp_reg(&ctx, instr.arg1), get_temp_reg(&ctx, instr.arg2));
            break;
        case OP_SUB:
            fprintf(out, "\tsub\tw%d, w%d, w%d\n", get_temp_reg(&ctx, instr.result), get_temp_reg(&ctx, instr.arg1), get_temp_reg(&ctx, instr.arg2));
            break;
        case OP_MUL:
            fprintf(out, "\tmul\tw%d, w%d, w%d\n", get_temp_reg(&ctx, instr.result), get_temp_reg(&ctx, instr.arg1), get_temp_reg(&ctx, instr.arg2));
            break;
        case OP_DIV:
            fprintf(out, "\tsdiv\tw%d, w%d, w%d\n", get_temp_reg(&ctx, instr.result), get_temp_reg(&ctx, instr.arg1), get_temp_reg(&ctx, instr.arg2));
            break;
        case OP_LABEL:
            fprintf(out, "%s:\n", instr.result);
            break;
        case OP_ARG:
            fprintf(out, "\tmov\tw%d, w%d\t\t\t; Set arg%d\n", current_arg_reg, get_temp_reg(&ctx, instr.arg1), current_arg_reg);
            current_arg_reg++;
            break;
        case OP_CALL:
            fprintf(out, "\tbl\t_%s\n", instr.arg1);
            current_arg_reg = 0;
            break;
        case OP_GET_RETVAL:
            fprintf(out, "\tmov\tw%d, w0\t\t\t; Get return value\n", get_temp_reg(&ctx, instr.result));
            break;
        case OP_RETURN:
            if (strlen(instr.arg1) > 0)
                fprintf(out, "\tmov\tw0, w%d\t\t\t; Set return value from %s\n", get_temp_reg(&ctx, instr.arg1), instr.arg1);
            // Epilogue
            fprintf(out, "\tldp\tx29, x30, [sp, #%d]\n", total_stack_alloc - 16);
            fprintf(out, "\tadd\tsp, sp, #%d\n", total_stack_alloc);
            fprintf(out, "\tret\n");
            break;
        default:
            break;
        }
    }
    fprintf(out, "\t.cfi_endproc\n");
    fprintf(out, "\t\t\t\t\t\t; -- End function %s\n", func_info->name);
}

// --- 主程式 ---
int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        fprintf(stderr, "用法: %s <ir_file> [output_file.s]\n", argv[0]);
        return 1;
    }
    const char *input_filename = argv[1];
    const char *output_filename = (argc > 2) ? argv[2] : "output.s";

    IR_Instruction ir_code[MAX_CODE_SIZE];
    int code_size = 0;
    FunctionInfo functions[MAX_FUNCTIONS];
    int func_count = 0;

    FILE *f = fopen(input_filename, "r");
    if (!f)
    {
        perror("無法開啟 IR 檔案");
        return 1;
    }

    char line[256];
    int reading_code = 0;
    while (fgets(line, sizeof(line), f))
    {
        line[strcspn(line, "\n")] = 0;
        if (strlen(line) == 0)
            continue;
        char p1[100] = {0}, p2[100] = {0}, p3[100] = {0}, p4[100] = {0};
        sscanf(line, "%s %s %s %s", p1, p2, p3, p4);
        if (strcmp(p1, "CODE_START") == 0)
        {
            reading_code = 1;
            continue;
        }

        if (!reading_code)
        {
            if (strcmp(p1, "FUNC_INFO") == 0)
            {
                FunctionInfo *info = &functions[func_count++];
                strcpy(info->name, p2);
                info->param_count = atoi(p3);
                const char *pos = strstr(line, p3) + strlen(p3);
                for (int i = 0; i < info->param_count; i++)
                {
                    sscanf(pos, " %s", info->params[i]);
                    pos = strstr(pos, info->params[i]) + strlen(info->params[i]);
                }
            }
        }
        else
        {
            IR_Instruction *i = &ir_code[code_size++];
            i->opcode = string_to_opcode(p1);
            strcpy(i->result, strcmp(p2, "_") == 0 ? "" : p2);
            strcpy(i->arg1, strcmp(p3, "_") == 0 ? "" : p3);
            strcpy(i->arg2, strcmp(p4, "_") == 0 ? "" : p4);
        }
    }
    fclose(f);

    FILE *out = fopen(output_filename, "w");
    if (!out)
    {
        perror("無法開啟輸出檔案");
        return 1;
    }

    fprintf(out, "\t.section\t__TEXT,__text,regular,pure_instructions\n");
    fprintf(out, "\t.build_version macos, 15, 0\tsdk_version 15, 2\n");

    for (int i = 0; i < code_size; i++)
    {
        if (ir_code[i].opcode == OP_FUNC_BEGIN)
        {
            FunctionInfo *current_func_info = find_function_info(functions, func_count, ir_code[i].result);
            if (current_func_info)
            {
                translate_function(out, current_func_info, ir_code, i);
            }
        }
    }

    fprintf(out, ".subsections_via_symbols\n");
    fclose(out);
    printf("組合語言已成功寫入到 %s\n", output_filename);

    return 0;
}