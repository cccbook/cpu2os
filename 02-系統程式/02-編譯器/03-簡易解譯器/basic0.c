// 本程式由 ccc 指揮 grok 3 撰寫
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_VARIABLES 100
#define MAX_NAME 32
#define MAX_LINE 256
#define MAX_LINES 1000
#define MAX_LABELS 100

typedef struct {
    char name[MAX_NAME];
    double value;
} Variable;

typedef struct {
    char name[MAX_NAME];
    int line_number;
} Label;

typedef struct {
    Variable vars[MAX_VARIABLES];
    int var_count;
    Label labels[MAX_LABELS];
    int label_count;
    char* program[MAX_LINES];
    int line_count;
} Interpreter;

void init_interpreter(Interpreter* interp) {
    interp->var_count = 0;
    interp->label_count = 0;
    interp->line_count = 0;
}

double* get_variable(Interpreter* interp, const char* name) {
    for (int i = 0; i < interp->var_count; i++) {
        if (strcmp(interp->vars[i].name, name) == 0) {
            return &interp->vars[i].value;
        }
    }
    if (interp->var_count < MAX_VARIABLES) {
        strncpy(interp->vars[interp->var_count].name, name, MAX_NAME);
        interp->vars[interp->var_count].value = 0;
        return &interp->vars[interp->var_count++].value;
    }
    return NULL;
}

char* skip_whitespace(char* str) {
    while (isspace(*str)) str++;
    return str;
}

double parse_number(char** str) {
    double result = 0;
    double fraction = 0;
    double divisor = 1;
    int sign = 1;

    *str = skip_whitespace(*str);
    
    if (**str == '-') {
        sign = -1;
        (*str)++;
    }
    
    while (isdigit(**str)) {
        result = result * 10 + (**str - '0');
        (*str)++;
    }
    
    if (**str == '.') {
        (*str)++;
        while (isdigit(**str)) {
            fraction = fraction * 10 + (**str - '0');
            divisor *= 10;
            (*str)++;
        }
    }
    
    return sign * (result + fraction / divisor);
}

char* parse_identifier(char* str, char* output) {
    str = skip_whitespace(str);
    int i = 0;
    while (isalnum(*str) || *str == '_') {
        if (i < MAX_NAME - 1) {
            output[i++] = *str;
        }
        str++;
    }
    output[i] = '\0';
    return str;
}

double evaluate_expression(Interpreter* interp, char** str);

double parse_factor(Interpreter* interp, char** str) {
    *str = skip_whitespace(*str);
    
    if (isdigit(**str) || **str == '-') {
        return parse_number(str);
    }
    
    char name[MAX_NAME];
    *str = parse_identifier(*str, name);
    double* var = get_variable(interp, name);
    return var ? *var : 0;
}

double evaluate_expression(Interpreter* interp, char** str) {
    double left = parse_factor(interp, str);
    
    while (1) {
        *str = skip_whitespace(*str);
        char op = **str;
        
        if (op == '<' || op == '>' || op == '=') {
            (*str)++;
            if (op == '=' && **str == '=') (*str)++;  // 處理 ==
            double right = parse_factor(interp, str);
            switch (op) {
                case '<': return left < right;
                case '>': return left > right;
                case '=': return left == right;
            }
        }
        else if (op != '+' && op != '-' && op != '*' && op != '/') {
            break;
        }
        else {
            (*str)++;
            double right = parse_factor(interp, str);
            switch (op) {
                case '+': left += right; break;
                case '-': left -= right; break;
                case '*': left *= right; break;
                case '/': left /= right; break;
            }
        }
    }
    
    return left;
}

int find_label(Interpreter* interp, const char* name) {
    for (int i = 0; i < interp->label_count; i++) {
        if (strcmp(interp->labels[i].name, name) == 0) {
            return interp->labels[i].line_number;
        }
    }
    return -1;
}

int execute_line(Interpreter* interp, char* line, int current_line) {
    line = skip_whitespace(line);
    
    // 標籤
    if (line[strlen(line) - 1] == ':') {
        char name[MAX_NAME];
        strncpy(name, line, strlen(line) - 1);
        name[strlen(line) - 1] = '\0';
        if (interp->label_count < MAX_LABELS) {
            strcpy(interp->labels[interp->label_count].name, name);
            interp->labels[interp->label_count].line_number = current_line;
            interp->label_count++;
        }
        return current_line + 1;
    }
    
    // goto
    if (strncmp(line, "goto ", 5) == 0) {
        char name[MAX_NAME];
        parse_identifier(line + 5, name);
        int target = find_label(interp, name);
        return (target != -1) ? target : current_line + 1;
    }
    
    // if
    if (strncmp(line, "if ", 3) == 0) {
        line += 3;
        double condition = evaluate_expression(interp, &line);
        line = skip_whitespace(line);
        if (strncmp(line, "goto ", 5) == 0 && condition) {
            char name[MAX_NAME];
            parse_identifier(line + 5, name);
            int target = find_label(interp, name);
            return (target != -1) ? target : current_line + 1;
        }
        return current_line + 1;
    }
    
    // 變數賦值
    char name[MAX_NAME];
    char* temp = parse_identifier(line, name);
    
    if (*temp == '=') {
        temp++;
        double* var = get_variable(interp, name);
        if (var) {
            *var = evaluate_expression(interp, &temp);
        }
    }
    // print
    else if (strncmp(line, "print", 5) == 0) {
        line += 5;
        double result = evaluate_expression(interp, &line);
        printf("%g\n", result);
    }
    
    return current_line + 1;
}

void run_program(Interpreter* interp) {
    int current_line = 0;
    while (current_line < interp->line_count) {
        current_line = execute_line(interp, interp->program[current_line], current_line);
    }
}

int main() {
    Interpreter interp;
    init_interpreter(&interp);
    char line[MAX_LINE];
    
    printf("Simple Basic-like Interpreter with GOTO and IF\n");
    printf("Enter program lines (empty line to run, 'quit' to exit):\n");
    
    while (1) {
        printf("> ");
        if (!fgets(line, sizeof(line), stdin)) break;
        
        line[strcspn(line, "\n")] = 0;
        
        if (strcmp(line, "quit") == 0) break;
        
        if (strlen(line) == 0) {
            run_program(&interp);
            init_interpreter(&interp);
            printf("\nEnter new program:\n");
            continue;
        }
        
        if (interp.line_count < MAX_LINES) {
            interp.program[interp.line_count] = strdup(line);
            interp.line_count++;
        }
    }
    
    for (int i = 0; i < interp.line_count; i++) {
        free(interp.program[i]);
    }
    
    return 0;
}