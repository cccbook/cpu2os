#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_TOKEN_LEN 100
#define MAX_VARS 100

// Token types
typedef enum {
    TOKEN_EOF,
    TOKEN_NUMBER,
    TOKEN_VARIABLE,
    TOKEN_OPERATOR,
    TOKEN_SEMICOLON,
    TOKEN_IF,
    TOKEN_WHILE,
    TOKEN_LPAREN,
    TOKEN_RPAREN,
    TOKEN_LBRACE,
    TOKEN_RBRACE,
    TOKEN_EQUALS,
    TOKEN_COMPARISON  // New token type for comparison operators
} TokenType;

// Token structure
typedef struct {
    TokenType type;
    char value[MAX_TOKEN_LEN];
} Token;

// Variable storage
typedef struct {
    char name[MAX_TOKEN_LEN];
    int value;
} Variable;

// Global variables
char *input;
int pos = 0;
Token current_token;
Variable variables[MAX_VARS];
int var_count = 0;

// Function declarations 
void get_next_token();
int parse_program();
int parse_statement();
int parse_block();
int parse_assignment();
int parse_expression();
int find_or_create_variable(const char *name);

// Get next token from input
void get_next_token() {
    // Skip whitespace
    while (isspace(input[pos])) pos++;
    
    if (input[pos] == '\0') {
        current_token.type = TOKEN_EOF;
        return;
    }
    
    // Numbers
    if (isdigit(input[pos])) {
        int i = 0;
        while (isdigit(input[pos])) {
            current_token.value[i++] = input[pos++];
        }
        current_token.value[i] = '\0';
        current_token.type = TOKEN_NUMBER;
        return;
    }
    
    // Variables and keywords
    if (isalpha(input[pos])) {
        int i = 0;
        while (isalnum(input[pos])) {
            current_token.value[i++] = input[pos++];
        }
        current_token.value[i] = '\0';
        
        if (strcmp(current_token.value, "if") == 0) {
            current_token.type = TOKEN_IF;
        } else if (strcmp(current_token.value, "while") == 0) {
            current_token.type = TOKEN_WHILE;
        } else {
            current_token.type = TOKEN_VARIABLE;
        }
        return;
    }
    
    // Single character tokens
    switch (input[pos]) {
        case '+': case '-': case '*': case '/':
            current_token.type = TOKEN_OPERATOR;
            current_token.value[0] = input[pos++];
            current_token.value[1] = '\0';
            break;
        case '<': case '>':  // Handle comparison operators
            current_token.type = TOKEN_COMPARISON;
            current_token.value[0] = input[pos++];
            current_token.value[1] = '\0';
            break;
        case ';':
            current_token.type = TOKEN_SEMICOLON;
            pos++;
            break;
        case '(':
            current_token.type = TOKEN_LPAREN;
            pos++;
            break;
        case ')':
            current_token.type = TOKEN_RPAREN;
            pos++;
            break;
        case '{':
            current_token.type = TOKEN_LBRACE;
            pos++;
            break;
        case '}':
            current_token.type = TOKEN_RBRACE;
            pos++;
            break;
        case '=':
            current_token.type = TOKEN_EQUALS;
            pos++;
            break;
        default:
            printf("Error: Unknown character '%c'\n", input[pos]);
            exit(1);
    }
}

// Find or create variable
int find_or_create_variable(const char *name) {
    for (int i = 0; i < var_count; i++) {
        if (strcmp(variables[i].name, name) == 0) {
            return i;
        }
    }
    
    strcpy(variables[var_count].name, name);
    variables[var_count].value = 0;
    return var_count++;
}

// Parse expression
int parse_expression() {
    int left;
    
    if (current_token.type == TOKEN_NUMBER) {
        left = atoi(current_token.value);
        get_next_token();
    } else if (current_token.type == TOKEN_VARIABLE) {
        int var_idx = find_or_create_variable(current_token.value);
        left = variables[var_idx].value;
        get_next_token();
    } else {
        printf("Error: Expected number or variable\n");
        exit(1);
    }
    
    while (current_token.type == TOKEN_OPERATOR || current_token.type == TOKEN_COMPARISON) {
        char op = current_token.value[0];
        get_next_token();
        
        int right;
        if (current_token.type == TOKEN_NUMBER) {
            right = atoi(current_token.value);
            get_next_token();
        } else if (current_token.type == TOKEN_VARIABLE) {
            int var_idx = find_or_create_variable(current_token.value);
            right = variables[var_idx].value;
            get_next_token();
        } else {
            printf("Error: Expected number or variable\n");
            exit(1);
        }
        
        switch (op) {
            case '+': left += right; break;
            case '-': left -= right; break;
            case '*': left *= right; break;
            case '/': 
                if (right == 0) {
                    printf("Error: Division by zero\n");
                    exit(1);
                }
                left /= right; 
                break;
            case '<': left = left < right; break;  // Added comparison operators
            case '>': left = left > right; break;
        }
    }
    
    return left;
}

// Parse assignment
int parse_assignment() {
    if (current_token.type != TOKEN_VARIABLE) {
        printf("Error: Expected variable name\n");
        exit(1);
    }
    
    int var_idx = find_or_create_variable(current_token.value);
    get_next_token();
    
    if (current_token.type != TOKEN_EQUALS) {
        printf("Error: Expected '='\n");
        exit(1);
    }
    get_next_token();
    
    int value = parse_expression();
    variables[var_idx].value = value;
    
    return value;
}

// Parse block
int parse_block() {
    if (current_token.type != TOKEN_LBRACE) {
        printf("Error: Expected '{'\n");
        exit(1);
    }
    get_next_token();
    
    int last_value = 0;
    while (current_token.type != TOKEN_RBRACE && current_token.type != TOKEN_EOF) {
        last_value = parse_statement();
    }
    
    if (current_token.type != TOKEN_RBRACE) {
        printf("Error: Expected '}'\n");
        exit(1);
    }
    get_next_token();
    
    return last_value;
}

// Parse statement
int parse_statement() {
    int value = 0;
    
    if (current_token.type == TOKEN_IF) {
        get_next_token();
        
        if (current_token.type != TOKEN_LPAREN) {
            printf("Error: Expected '('\n");
            exit(1);
        }
        get_next_token();
        
        int condition = parse_expression();
        
        if (current_token.type != TOKEN_RPAREN) {
            printf("Error: Expected ')'\n");
            exit(1);
        }
        get_next_token();
        
        if (condition) {
            value = parse_block();
        } else {
            // Skip the block
            parse_block();
        }
    } else if (current_token.type == TOKEN_WHILE) {
        get_next_token();
        
        if (current_token.type != TOKEN_LPAREN) {
            printf("Error: Expected '('\n");
            exit(1);
        }
        get_next_token();
        
        int start_pos = pos;
        Token start_token = current_token;
        
        int condition = parse_expression();
        
        if (current_token.type != TOKEN_RPAREN) {
            printf("Error: Expected ')'\n");
            exit(1);
        }
        get_next_token();
        
        while (condition) {
            value = parse_block();
            
            pos = start_pos;
            current_token = start_token;
            condition = parse_expression();
            
            if (current_token.type != TOKEN_RPAREN) {
                printf("Error: Expected ')'\n");
                exit(1);
            }
            get_next_token();
        }
        // Skip the block if condition is false
        parse_block();
    } else {
        value = parse_assignment();
        
        if (current_token.type != TOKEN_SEMICOLON) {
            printf("Error: Expected ';'\n");
            exit(1);
        }
        get_next_token();
    }
    
    return value;
}

// Parse program
int parse_program() {
    int last_value = 0;
    get_next_token();
    
    while (current_token.type != TOKEN_EOF) {
        last_value = parse_statement();
    }
    
    return last_value;
}

// Main function
int main() {
    // Example program
    char program[] = 
        "x = 5;\n"
        "y = 10;\n"
        "if (x < y) {\n"
        "    z = x + y;\n"
        "}\n"
        "while (x > 0) {\n"
        "    x = x - 1;\n"
        "    w = y + z;\n"
        "}\n";
    
    input = program;
    int result = parse_program();
    
    // Print all variables
    printf("Final variable values:\n");
    for (int i = 0; i < var_count; i++) {
        printf("%s = %d\n", variables[i].name, variables[i].value);
    }
    
    return 0;
}