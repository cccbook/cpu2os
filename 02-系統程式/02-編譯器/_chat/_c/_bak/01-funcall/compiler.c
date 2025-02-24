#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_TOKEN_LEN 100
#define MAX_FUNCTIONS 100
#define MAX_PARAMS 10

// 函數表結構
typedef struct {
    char name[MAX_TOKEN_LEN];
    int paramCount;
    int address;
} Function;

// 全域變數
Function functionTable[MAX_FUNCTIONS];
int functionCount = 0;
char currentToken[MAX_TOKEN_LEN];
FILE* inputFile;
int currentAddress = 0;

// 詞法分析器
void getNextToken() {
    int i = 0;
    char c;
    
    // 跳過空白字元
    while ((c = fgetc(inputFile)) != EOF && isspace(c));
    
    if (c == EOF) {
        currentToken[0] = '\0';
        return;
    }
    
    // 讀取標識符或關鍵字
    if (isalpha(c)) {
        currentToken[i++] = c;
        while ((c = fgetc(inputFile)) != EOF && (isalnum(c) || c == '_')) {
            currentToken[i++] = c;
        }
        ungetc(c, inputFile);
        currentToken[i] = '\0';
        return;
    }
    
    // 處理特殊字元
    currentToken[0] = c;
    currentToken[1] = '\0';
}

// 添加函數到函數表
void addFunction(const char* name, int paramCount) {
    if (functionCount >= MAX_FUNCTIONS) {
        fprintf(stderr, "Error: Too many functions\n");
        exit(1);
    }
    
    strcpy(functionTable[functionCount].name, name);
    functionTable[functionCount].paramCount = paramCount;
    functionTable[functionCount].address = currentAddress;
    functionCount++;
    currentAddress += 1; // 簡化版：每個函數佔用1個位址
}

// 尋找函數
Function* findFunction(const char* name) {
    for (int i = 0; i < functionCount; i++) {
        if (strcmp(functionTable[i].name, name) == 0) {
            return &functionTable[i];
        }
    }
    return NULL;
}

// 解析函數定義
void parseFunction() {
    getNextToken(); // 讀取函數名稱
    char functionName[MAX_TOKEN_LEN];
    strcpy(functionName, currentToken);
    
    getNextToken(); // 讀取左括號
    if (strcmp(currentToken, "(") != 0) {
        fprintf(stderr, "Error: Expected '('\n");
        exit(1);
    }
    
    // 計算參數數量
    int paramCount = 0;
    getNextToken();
    while (strcmp(currentToken, ")") != 0) {
        paramCount++;
        getNextToken(); // 讀取逗號或右括號
        if (strcmp(currentToken, ",") == 0) {
            getNextToken(); // 讀取下一個參數
        }
    }
    
    // 添加函數到函數表
    addFunction(functionName, paramCount);
    
    // 讀取函數體
    getNextToken(); // 讀取左大括號
    while (strcmp(currentToken, "}") != 0) {
        getNextToken();
    }
}

// 解析函數呼叫
void parseFunctionCall() {
    char functionName[MAX_TOKEN_LEN];
    strcpy(functionName, currentToken);
    
    Function* func = findFunction(functionName);
    if (func == NULL) {
        fprintf(stderr, "Error: Undefined function '%s'\n", functionName);
        exit(1);
    }
    
    getNextToken(); // 讀取左括號
    if (strcmp(currentToken, "(") != 0) {
        fprintf(stderr, "Error: Expected '('\n");
        exit(1);
    }
    
    // 計算並檢查參數數量
    int paramCount = 0;
    getNextToken();
    while (strcmp(currentToken, ")") != 0) {
        paramCount++;
        getNextToken(); // 讀取逗號或右括號
        if (strcmp(currentToken, ",") == 0) {
            getNextToken(); // 讀取下一個參數
        }
    }
    
    if (paramCount != func->paramCount) {
        fprintf(stderr, "Error: Wrong number of parameters for function '%s'\n", functionName);
        exit(1);
    }
    
    // 生成函數呼叫指令（簡化版）
    printf("CALL %s\n", functionName);
}

// 主要解析函式
void parse() {
    while (1) {
        getNextToken();
        if (strlen(currentToken) == 0) break;
        
        if (strcmp(currentToken, "function") == 0) {
            parseFunction();
        } else if (findFunction(currentToken) != NULL) {
            parseFunctionCall();
        } else {
            fprintf(stderr, "Error: Unexpected token '%s'\n", currentToken);
            exit(1);
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
        return 1;
    }
    
    inputFile = fopen(argv[1], "r");
    if (inputFile == NULL) {
        fprintf(stderr, "Error: Cannot open input file\n");
        return 1;
    }
    
    parse();
    
    fclose(inputFile);
    return 0;
}