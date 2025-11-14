#include "ir.h"

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

char *opcode_to_string(OpCode opcode)
{
    char *opcode_str = "";
    switch (opcode)
    {
    case OP_ADD:
        opcode_str = "OP_ADD";
        break;
    case OP_SUB:
        opcode_str = "OP_SUB";
        break;
    case OP_MUL:
        opcode_str = "OP_MUL";
        break;
    case OP_DIV:
        opcode_str = "OP_DIV";
        break;
    case OP_EQ:
        opcode_str = "OP_EQ";
        break;
    case OP_NE:
        opcode_str = "OP_NE";
        break;
    case OP_LT:
        opcode_str = "OP_LT";
        break;
    case OP_GT:
        opcode_str = "OP_GT";
        break;
    case OP_LOAD_CONST:
        opcode_str = "OP_LOAD_CONST";
        break;
    case OP_LOAD_VAR:
        opcode_str = "OP_LOAD_VAR";
        break;
    case OP_STORE_VAR:
        opcode_str = "OP_STORE_VAR";
        break;
    case OP_GOTO:
        opcode_str = "OP_GOTO";
        break;
    case OP_IF_FALSE_GOTO:
        opcode_str = "OP_IF_FALSE_GOTO";
        break;
    case OP_LABEL:
        opcode_str = "OP_LABEL";
        break;
    case OP_FUNC_BEGIN:
        opcode_str = "OP_FUNC_BEGIN";
        break;
    case OP_FUNC_END:
        opcode_str = "OP_FUNC_END";
        break;
    case OP_CALL:
        opcode_str = "OP_CALL";
        break;
    case OP_ARG:
        opcode_str = "OP_ARG";
        break;
    case OP_RETURN:
        opcode_str = "OP_RETURN";
        break;
    case OP_GET_RETVAL:
        opcode_str = "OP_GET_RETVAL";
        break;
    case OP_UNKNOWN:
        opcode_str = "OP_UNKNOWN";
        break;
    }
    return opcode_str;
}