#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_CODE_SIZE 1024
#define MAX_FUNCTIONS 50
#define MAX_LABELS 200
#define MAX_LOCALS 100
#define MAX_CALL_STACK 100
#define MAX_ARGS 10

// --- IR/Bytecode 定義 (與 compiler.c 中的定義一致) ---
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
    OP_STORE_VAR,//10
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

OpCode string_to_opcode(const char *s);
const char *opcode_to_string(OpCode opcode);
