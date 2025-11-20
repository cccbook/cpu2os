#include "ir.h"

// 將字串 op code 轉換回 enum
OpCode string_to_opcode(const char *s)
{
    if (strcmp(s, "ADD") == 0)
        return OP_ADD;
    if (strcmp(s, "SUB") == 0)
        return OP_SUB;
    if (strcmp(s, "MUL") == 0)
        return OP_MUL;
    if (strcmp(s, "DIV") == 0)
        return OP_DIV;
    if (strcmp(s, "EQ") == 0)
        return OP_EQ;
    if (strcmp(s, "NE") == 0)
        return OP_NE;
    if (strcmp(s, "LT") == 0)
        return OP_LT;
    if (strcmp(s, "GT") == 0)
        return OP_GT;
    if (strcmp(s, "LOAD_CONST") == 0)
        return OP_LOAD_CONST;
    if (strcmp(s, "LOAD_VAR") == 0)
        return OP_LOAD_VAR;
    if (strcmp(s, "STORE_VAR") == 0)
        return OP_STORE_VAR;
    if (strcmp(s, "GOTO") == 0)
        return OP_GOTO;
    if (strcmp(s, "IF_FALSE_GOTO") == 0)
        return OP_IF_FALSE_GOTO;
    if (strcmp(s, "LABEL") == 0)
        return OP_LABEL;
    if (strcmp(s, "FUNC_BEGIN") == 0)
        return OP_FUNC_BEGIN;
    if (strcmp(s, "FUNC_END") == 0)
        return OP_FUNC_END;
    if (strcmp(s, "CALL") == 0)
        return OP_CALL;
    if (strcmp(s, "ARG") == 0)
        return OP_ARG;
    if (strcmp(s, "RETURN") == 0)
        return OP_RETURN;
    if (strcmp(s, "GET_RETVAL") == 0)
        return OP_GET_RETVAL;
    return OP_UNKNOWN;
}

const char *opcode_to_string(OpCode opcode)
{
    char *opcode_str = "";
    switch (opcode)
    {
    case OP_ADD:
        opcode_str = "ADD";
        break;
    case OP_SUB:
        opcode_str = "SUB";
        break;
    case OP_MUL:
        opcode_str = "MUL";
        break;
    case OP_DIV:
        opcode_str = "DIV";
        break;
    case OP_EQ:
        opcode_str = "EQ";
        break;
    case OP_NE:
        opcode_str = "NE";
        break;
    case OP_LT:
        opcode_str = "LT";
        break;
    case OP_GT:
        opcode_str = "GT";
        break;
    case OP_LOAD_CONST:
        opcode_str = "LOAD_CONST";
        break;
    case OP_LOAD_VAR:
        opcode_str = "LOAD_VAR";
        break;
    case OP_STORE_VAR:
        opcode_str = "STORE_VAR";
        break;
    case OP_GOTO:
        opcode_str = "GOTO";
        break;
    case OP_IF_FALSE_GOTO:
        opcode_str = "IF_FALSE_GOTO";
        break;
    case OP_LABEL:
        opcode_str = "LABEL";
        break;
    case OP_FUNC_BEGIN:
        opcode_str = "FUNC_BEGIN";
        break;
    case OP_FUNC_END:
        opcode_str = "FUNC_END";
        break;
    case OP_CALL:
        opcode_str = "CALL";
        break;
    case OP_ARG:
        opcode_str = "ARG";
        break;
    case OP_RETURN:
        opcode_str = "RETURN";
        break;
    case OP_GET_RETVAL:
        opcode_str = "GET_RETVAL";
        break;
    case OP_UNKNOWN:
        opcode_str = "UNKNOWN";
        break;
    }
    return opcode_str;
}