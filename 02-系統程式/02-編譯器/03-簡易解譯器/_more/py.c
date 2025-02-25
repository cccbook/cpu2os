#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <unistd.h>  // 提供 ftruncate
#include <fcntl.h>   // 提供 fileno

#define MAX_TOKENS 100
#define MAX_TOKEN_LEN 100
#define MAX_VAR_NAME 50
#define MAX_VARIABLES 100
#define MAX_INDENT_LEVELS 10

// 變量結構
typedef struct {
    char name[MAX_VAR_NAME];
    double value;
} Variable;

// 變量存儲
Variable variables[MAX_VARIABLES];
int var_count = 0;

// 字符串處理相關函數
char* trim(char* str) {
    char* end;
    // 去掉前置空格
    while(isspace((unsigned char)*str)) str++;
    
    if(*str == 0) return str;  // 全是空格
    
    // 去掉後置空格
    end = str + strlen(str) - 1;
    while(end > str && isspace((unsigned char)*end)) end--;
    
    end[1] = '\0';  // 添加結束符
    
    return str;
}

// 分割行為標記
int tokenize(char* line, char tokens[MAX_TOKENS][MAX_TOKEN_LEN]) {
    int count = 0;
    char* token = strtok(line, " \t\n");
    
    while(token != NULL && count < MAX_TOKENS) {
        strcpy(tokens[count++], token);
        token = strtok(NULL, " \t\n");
    }
    
    return count;
}

// 查找或創建變量
int find_or_create_var(const char* name) {
    // 先查找
    for(int i = 0; i < var_count; i++) {
        if(strcmp(variables[i].name, name) == 0) {
            return i;
        }
    }
    
    // 不存在則創建
    if(var_count < MAX_VARIABLES) {
        strcpy(variables[var_count].name, name);
        variables[var_count].value = 0;
        return var_count++;
    }
    
    fprintf(stderr, "錯誤: 變量數量超出限制\n");
    return -1;
}

// 獲取變量值或將字符串轉為數字
double get_value(const char* token) {
    // 嘗試作為變量名查找
    for(int i = 0; i < var_count; i++) {
        if(strcmp(variables[i].name, token) == 0) {
            return variables[i].value;
        }
    }
    
    // 嘗試轉換為數字
    char* endptr;
    double value = strtod(token, &endptr);
    
    // 檢查轉換是否成功
    if(*endptr == '\0') {
        return value;
    }
    
    // 找不到變量且不是數字，創建新變量
    int index = find_or_create_var(token);
    if(index >= 0) {
        return variables[index].value;
    }
    
    return 0;
}

// 執行賦值操作
void execute_assignment(const char* var_name, const char* op, const char* value_str) {
    int var_index = find_or_create_var(var_name);
    if(var_index < 0) return;
    
    double value = get_value(value_str);
    
    // 處理不同賦值操作符
    if(strcmp(op, "=") == 0) {
        variables[var_index].value = value;
    } else if(strcmp(op, "+=") == 0) {
        variables[var_index].value += value;
    } else if(strcmp(op, "-=") == 0) {
        variables[var_index].value -= value;
    } else if(strcmp(op, "*=") == 0) {
        variables[var_index].value *= value;
    } else if(strcmp(op, "/=") == 0) {
        if(value == 0) {
            fprintf(stderr, "錯誤: 除數不能為零\n");
            return;
        }
        variables[var_index].value /= value;
    } else {
        fprintf(stderr, "錯誤: 不支持的操作符 '%s'\n", op);
    }
}

// 計算表達式值
double evaluate_expression(char tokens[MAX_TOKENS][MAX_TOKEN_LEN], int start, int end) {
    if(start > end) return 0;
    if(start == end) return get_value(tokens[start]);
    
    // 查找優先級最低的操作符
    int op_index = -1;
    int paren_level = 0;
    int min_priority = 99;
    
    for(int i = start; i <= end; i++) {
        if(strcmp(tokens[i], "(") == 0) {
            paren_level++;
        } else if(strcmp(tokens[i], ")") == 0) {
            paren_level--;
        } else if(paren_level == 0) {
            int priority;
            if(strcmp(tokens[i], "+") == 0 || strcmp(tokens[i], "-") == 0) {
                priority = 1;
            } else if(strcmp(tokens[i], "*") == 0 || strcmp(tokens[i], "/") == 0) {
                priority = 2;
            } else if(strcmp(tokens[i], "**") == 0) {
                priority = 3;
            } else {
                priority = 99;  // 不是操作符
            }
            
            if(priority < min_priority) {
                min_priority = priority;
                op_index = i;
            }
        }
    }
    
    // 如果沒找到操作符，可能是括號表達式
    if(op_index == -1) {
        if(strcmp(tokens[start], "(") == 0 && strcmp(tokens[end], ")") == 0) {
            return evaluate_expression(tokens, start + 1, end - 1);
        }
        return get_value(tokens[start]);
    }
    
    // 計算操作符左右兩側表達式的值
    double left = evaluate_expression(tokens, start, op_index - 1);
    double right = evaluate_expression(tokens, op_index + 1, end);
    
    // 執行對應操作
    if(strcmp(tokens[op_index], "+") == 0) {
        return left + right;
    } else if(strcmp(tokens[op_index], "-") == 0) {
        return left - right;
    } else if(strcmp(tokens[op_index], "*") == 0) {
        return left * right;
    } else if(strcmp(tokens[op_index], "/") == 0) {
        if(right == 0) {
            fprintf(stderr, "錯誤: 除數不能為零\n");
            return 0;
        }
        return left / right;
    } else if(strcmp(tokens[op_index], "**") == 0) {
        return pow(left, right);
    }
    
    return 0;
}

// 執行print語句
void execute_print(char tokens[MAX_TOKENS][MAX_TOKEN_LEN], int count) {
    for(int i = 1; i < count; i++) {
        // 處理字符串（被雙引號包圍）
        if(tokens[i][0] == '"' && tokens[i][strlen(tokens[i])-1] == '"') {
            // 去掉引號
            tokens[i][strlen(tokens[i])-1] = '\0';
            printf("%s ", tokens[i] + 1);
        } else {
            // 打印數值
            printf("%.6g ", get_value(tokens[i]));
        }
    }
    printf("\n");
}

// 執行if語句
int execute_if(char tokens[MAX_TOKENS][MAX_TOKEN_LEN], int count) {
    if(count < 4 || strcmp(tokens[count-1], ":") != 0) {
        fprintf(stderr, "錯誤: if 語句格式不正確\n");
        return 0;
    }
    
    // 查找比較操作符
    int op_index = -1;
    for(int i = 1; i < count - 1; i++) {
        if(strcmp(tokens[i], "==") == 0 || 
           strcmp(tokens[i], "!=") == 0 ||
           strcmp(tokens[i], ">") == 0 ||
           strcmp(tokens[i], "<") == 0 ||
           strcmp(tokens[i], ">=") == 0 ||
           strcmp(tokens[i], "<=") == 0) {
            op_index = i;
            break;
        }
    }
    
    if(op_index == -1) {
        // 單一條件
        double value = get_value(tokens[1]);
        return value != 0;
    }
    
    // 比較條件
    double left = get_value(tokens[op_index - 1]);
    double right = get_value(tokens[op_index + 1]);
    
    if(strcmp(tokens[op_index], "==") == 0) {
        return left == right;
    } else if(strcmp(tokens[op_index], "!=") == 0) {
        return left != right;
    } else if(strcmp(tokens[op_index], ">") == 0) {
        return left > right;
    } else if(strcmp(tokens[op_index], "<") == 0) {
        return left < right;
    } else if(strcmp(tokens[op_index], ">=") == 0) {
        return left >= right;
    } else if(strcmp(tokens[op_index], "<=") == 0) {
        return left <= right;
    }
    
    return 0;
}

// 主要解譯函數
void interpret(FILE* file) {
    char line[1024];
    char tokens[MAX_TOKENS][MAX_TOKEN_LEN];
    int current_indent = 0;
    int indent_levels[MAX_INDENT_LEVELS] = {0};
    int indent_count = 1; // 首層的縮進級別為0
    int executing_block = 1; // 是否執行當前代碼塊
    int if_result = 0; // 上一個if語句的結果
    
    while(fgets(line, sizeof(line), file)) {
        char* trimmed = trim(line);
        
        // 跳過空行和注釋
        if(strlen(trimmed) == 0 || trimmed[0] == '#') {
            continue;
        }
        
        // 計算縮進
        int indent = 0;
        char* line_start = line;
        while(*line_start == ' ' || *line_start == '\t') {
            indent++;
            line_start++;
        }
        
        // 處理縮進變化
        if(indent > current_indent) {
            // 增加縮進
            if(indent_count < MAX_INDENT_LEVELS) {
                indent_levels[indent_count] = indent;
                current_indent = indent;
                indent_count++;
                
                // 如果前一個if語句結果為假，不執行當前代碼塊
                if(!if_result) {
                    executing_block = 0;
                }
            } else {
                fprintf(stderr, "錯誤: 縮進層級過多\n");
                return;
            }
        } else if(indent < current_indent) {
            // 減少縮進
            while(indent_count > 1 && indent < indent_levels[indent_count - 1]) {
                indent_count--;
            }
            
            if(indent_count > 0) {
                current_indent = indent_levels[indent_count - 1];
                executing_block = 1; // 退出代碼塊時恢復執行
            }
        }
        
        // 如果不在執行的代碼塊中，跳過此行
        if(!executing_block && indent > indent_levels[indent_count - 2]) {
            continue;
        }
        
        // 處理語句
        char line_copy[1024];
        strcpy(line_copy, trimmed);
        int token_count = tokenize(line_copy, tokens);
        
        if(token_count == 0) continue;
        
        // 處理不同類型的語句
        if(strcmp(tokens[0], "print") == 0) {
            execute_print(tokens, token_count);
        } else if(strcmp(tokens[0], "if") == 0) {
            if_result = execute_if(tokens, token_count);
        } else if(token_count >= 3 && (
                  strcmp(tokens[1], "=") == 0 ||
                  strcmp(tokens[1], "+=") == 0 ||
                  strcmp(tokens[1], "-=") == 0 ||
                  strcmp(tokens[1], "*=") == 0 ||
                  strcmp(tokens[1], "/=") == 0)) {
            // 賦值語句
            execute_assignment(tokens[0], tokens[1], tokens[2]);
        } else {
            // 嘗試計算表達式
            double result = evaluate_expression(tokens, 0, token_count - 1);
            printf("表達式結果: %.6g\n", result);
        }
    }
}

int main() {
    printf("簡易Python風格解譯器 (C語言實現)\n");
    printf("輸入 'exit()' 結束程序\n\n");
    
    // 從標準輸入讀取代碼
    char input[1024];
    FILE* temp_file = tmpfile();
    
    if(!temp_file) {
        fprintf(stderr, "錯誤: 無法創建臨時文件\n");
        return 1;
    }
    
    while(1) {
        printf(">>> ");
        if(fgets(input, sizeof(input), stdin) == NULL) break;
        
        // 檢查退出命令
        if(strcmp(trim(input), "exit()") == 0) {
            break;
        }
        
        // 寫入臨時文件
        fputs(input, temp_file);
        
        // 檢查是否以冒號結尾，需要繼續輸入
        char* trimmed = trim(input);
        int len = strlen(trimmed);
        if(len > 0 && trimmed[len-1] == ':') {
            char indent_input[1024];
            while(1) {
                printf("... ");
                if(fgets(indent_input, sizeof(indent_input), stdin) == NULL) break;
                
                // 空行結束當前代碼塊
                if(strlen(trim(indent_input)) == 0) {
                    break;
                }
                
                fputs(indent_input, temp_file);
            }
        }
        
        // 解譯代碼
        fseek(temp_file, 0, SEEK_SET);
        interpret(temp_file);
        
        // 清空臨時文件以便下次使用
        ftruncate(fileno(temp_file), 0);
        fseek(temp_file, 0, SEEK_SET);
    }
    
    fclose(temp_file);
    return 0;
}