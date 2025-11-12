#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_LINE 256
#define MAX_TOKEN 128
#define MAX_SYMBOLS 1000

// Token types
typedef enum {
    TK_KEYWORD, TK_SYMBOL, TK_IDENTIFIER, TK_INT_CONST, 
    TK_STRING_CONST, TK_EOF
} TokenType;

// Keywords
typedef enum {
    KW_CLASS, KW_CONSTRUCTOR, KW_FUNCTION, KW_METHOD,
    KW_FIELD, KW_STATIC, KW_VAR, KW_INT, KW_CHAR,
    KW_BOOLEAN, KW_VOID, KW_TRUE, KW_FALSE, KW_NULL,
    KW_THIS, KW_LET, KW_DO, KW_IF, KW_ELSE, KW_WHILE, KW_RETURN
} Keyword;

// Symbol table entry
typedef struct {
    char name[MAX_TOKEN];
    char type[MAX_TOKEN];
    char kind[16]; // static, field, arg, var
    int index;
} Symbol;

// Symbol table
typedef struct {
    Symbol symbols[MAX_SYMBOLS];
    int count;
} SymbolTable;

// Compiler state
typedef struct {
    FILE *input;
    FILE *output;
    char token[MAX_TOKEN];
    TokenType tokenType;
    char className[MAX_TOKEN];
    SymbolTable classTable;
    SymbolTable subroutineTable;
    int labelCount;
    int fieldCount;
} Compiler;

// Keywords lookup
const char *keywords[] = {
    "class", "constructor", "function", "method", "field", "static",
    "var", "int", "char", "boolean", "void", "true", "false",
    "null", "this", "let", "do", "if", "else", "while", "return"
};

// Function prototypes
void initCompiler(Compiler *c, FILE *in, FILE *out);
int isKeyword(const char *str);
void advance(Compiler *c);
void compileClass(Compiler *c);
void compileClassVarDec(Compiler *c);
void compileSubroutine(Compiler *c);
void compileParameterList(Compiler *c);
void compileVarDec(Compiler *c);
void compileStatements(Compiler *c);
void compileLet(Compiler *c);
void compileIf(Compiler *c);
void compileWhile(Compiler *c);
void compileDo(Compiler *c);
void compileReturn(Compiler *c);
void compileExpression(Compiler *c);
void compileTerm(Compiler *c);
int compileExpressionList(Compiler *c);
void addSymbol(SymbolTable *table, const char *name, const char *type, const char *kind);
Symbol* findSymbol(Compiler *c, const char *name);

void initCompiler(Compiler *c, FILE *in, FILE *out) {
    c->input = in;
    c->output = out;
    c->classTable.count = 0;
    c->subroutineTable.count = 0;
    c->labelCount = 0;
    c->fieldCount = 0;
}

int isKeyword(const char *str) {
    for (int i = 0; i < 22; i++) {
        if (strcmp(str, keywords[i]) == 0) return 1;
    }
    return 0;
}

void skipWhitespaceAndComments(Compiler *c) {
    int ch;
    while ((ch = fgetc(c->input)) != EOF) {
        if (isspace(ch)) continue;
        if (ch == '/') {
            int next = fgetc(c->input);
            if (next == '/') {
                while ((ch = fgetc(c->input)) != EOF && ch != '\n');
            } else if (next == '*') {
                int prev = 0;
                while ((ch = fgetc(c->input)) != EOF) {
                    if (prev == '*' && ch == '/') break;
                    prev = ch;
                }
            } else {
                ungetc(next, c->input);
                ungetc(ch, c->input);
                return;
            }
        } else {
            ungetc(ch, c->input);
            return;
        }
    }
}

void advance(Compiler *c) {
    skipWhitespaceAndComments(c);
    int ch = fgetc(c->input);
    
    if (ch == EOF) {
        c->tokenType = TK_EOF;
        return;
    }
    
    // String constant
    if (ch == '"') {
        int i = 0;
        while ((ch = fgetc(c->input)) != '"' && ch != EOF) {
            c->token[i++] = ch;
        }
        c->token[i] = '\0';
        c->tokenType = TK_STRING_CONST;
        return;
    }
    
    // Symbol
    if (strchr("{}()[].,;+-*/&|<>=~", ch)) {
        c->token[0] = ch;
        c->token[1] = '\0';
        c->tokenType = TK_SYMBOL;
        return;
    }
    
    // Number
    if (isdigit(ch)) {
        int i = 0;
        c->token[i++] = ch;
        while (isdigit(ch = fgetc(c->input))) {
            c->token[i++] = ch;
        }
        ungetc(ch, c->input);
        c->token[i] = '\0';
        c->tokenType = TK_INT_CONST;
        return;
    }
    
    // Identifier or keyword
    if (isalpha(ch) || ch == '_') {
        int i = 0;
        c->token[i++] = ch;
        while (isalnum(ch = fgetc(c->input)) || ch == '_') {
            c->token[i++] = ch;
        }
        ungetc(ch, c->input);
        c->token[i] = '\0';
        c->tokenType = isKeyword(c->token) ? TK_KEYWORD : TK_IDENTIFIER;
        return;
    }
}

void addSymbol(SymbolTable *table, const char *name, const char *type, const char *kind) {
    Symbol *s = &table->symbols[table->count];
    strcpy(s->name, name);
    strcpy(s->type, type);
    strcpy(s->kind, kind);
    s->index = table->count;
    table->count++;
}

Symbol* findSymbol(Compiler *c, const char *name) {
    for (int i = 0; i < c->subroutineTable.count; i++) {
        if (strcmp(c->subroutineTable.symbols[i].name, name) == 0) {
            return &c->subroutineTable.symbols[i];
        }
    }
    for (int i = 0; i < c->classTable.count; i++) {
        if (strcmp(c->classTable.symbols[i].name, name) == 0) {
            return &c->classTable.symbols[i];
        }
    }
    return NULL;
}

void compileClass(Compiler *c) {
    advance(c); // class
    advance(c); // className
    strcpy(c->className, c->token);
    advance(c); // {
    
    while (1) {
        advance(c);
        if (strcmp(c->token, "static") == 0 || strcmp(c->token, "field") == 0) {
            compileClassVarDec(c);
        } else if (strcmp(c->token, "constructor") == 0 || 
                   strcmp(c->token, "function") == 0 || 
                   strcmp(c->token, "method") == 0) {
            compileSubroutine(c);
        } else {
            break; // }
        }
    }
}

void compileClassVarDec(Compiler *c) {
    char kind[16], type[MAX_TOKEN];
    strcpy(kind, c->token); // static or field
    if (strcmp(kind, "field") == 0) c->fieldCount++;
    
    advance(c); // type
    strcpy(type, c->token);
    
    advance(c); // varName
    addSymbol(&c->classTable, c->token, type, kind);
    
    advance(c);
    while (strcmp(c->token, ",") == 0) {
        advance(c); // varName
        addSymbol(&c->classTable, c->token, type, kind);
        if (strcmp(kind, "field") == 0) c->fieldCount++;
        advance(c);
    }
}

void compileSubroutine(Compiler *c) {
    char subroutineType[MAX_TOKEN];
    strcpy(subroutineType, c->token);
    
    c->subroutineTable.count = 0;
    int nArgs = 0;
    
    if (strcmp(subroutineType, "method") == 0) {
        addSymbol(&c->subroutineTable, "this", c->className, "arg");
        nArgs = 1;
    }
    
    advance(c); // return type
    advance(c); // subroutineName
    char subroutineName[MAX_TOKEN];
    sprintf(subroutineName, "%s.%s", c->className, c->token);
    
    advance(c); // (
    advance(c);
    compileParameterList(c);
    advance(c); // )
    
    advance(c); // {
    int nVars = 0;
    while (1) {
        advance(c);
        if (strcmp(c->token, "var") == 0) {
            int before = c->subroutineTable.count;
            compileVarDec(c);
            nVars += c->subroutineTable.count - before;
        } else {
            break;
        }
    }
    
    fprintf(c->output, "function %s %d\n", subroutineName, nVars);
    
    if (strcmp(subroutineType, "constructor") == 0) {
        fprintf(c->output, "push constant %d\n", c->fieldCount);
        fprintf(c->output, "call Memory.alloc 1\n");
        fprintf(c->output, "pop pointer 0\n");
    } else if (strcmp(subroutineType, "method") == 0) {
        fprintf(c->output, "push argument 0\n");
        fprintf(c->output, "pop pointer 0\n");
    }
    
    compileStatements(c);
    advance(c); // }
}

void compileParameterList(Compiler *c) {
    if (strcmp(c->token, ")") == 0) return;
    
    char type[MAX_TOKEN];
    strcpy(type, c->token);
    advance(c); // varName
    addSymbol(&c->subroutineTable, c->token, type, "arg");
    
    advance(c);
    while (strcmp(c->token, ",") == 0) {
        advance(c); // type
        strcpy(type, c->token);
        advance(c); // varName
        addSymbol(&c->subroutineTable, c->token, type, "arg");
        advance(c);
    }
}

void compileVarDec(Compiler *c) {
    char type[MAX_TOKEN];
    advance(c); // type
    strcpy(type, c->token);
    
    advance(c); // varName
    addSymbol(&c->subroutineTable, c->token, type, "var");
    
    advance(c);
    while (strcmp(c->token, ",") == 0) {
        advance(c);
        addSymbol(&c->subroutineTable, c->token, type, "var");
        advance(c);
    }
}

void compileStatements(Compiler *c) {
    while (1) {
        if (strcmp(c->token, "let") == 0) compileLet(c);
        else if (strcmp(c->token, "if") == 0) compileIf(c);
        else if (strcmp(c->token, "while") == 0) compileWhile(c);
        else if (strcmp(c->token, "do") == 0) compileDo(c);
        else if (strcmp(c->token, "return") == 0) compileReturn(c);
        else break;
    }
}

void compileLet(Compiler *c) {
    advance(c); // varName
    char varName[MAX_TOKEN];
    strcpy(varName, c->token);
    Symbol *sym = findSymbol(c, varName);
    
    advance(c);
    int isArray = (strcmp(c->token, "[") == 0);
    
    if (isArray) {
        advance(c);
        compileExpression(c);
        advance(c); // ]
        
        fprintf(c->output, "push %s %d\n", 
                strcmp(sym->kind, "var") == 0 ? "local" : 
                strcmp(sym->kind, "arg") == 0 ? "argument" :
                strcmp(sym->kind, "field") == 0 ? "this" : "static",
                sym->index);
        fprintf(c->output, "add\n");
        
        advance(c); // =
        advance(c);
        compileExpression(c);
        
        fprintf(c->output, "pop temp 0\n");
        fprintf(c->output, "pop pointer 1\n");
        fprintf(c->output, "push temp 0\n");
        fprintf(c->output, "pop that 0\n");
    } else {
        advance(c); // =
        advance(c);
        compileExpression(c);
        
        fprintf(c->output, "pop %s %d\n", 
                strcmp(sym->kind, "var") == 0 ? "local" : 
                strcmp(sym->kind, "arg") == 0 ? "argument" :
                strcmp(sym->kind, "field") == 0 ? "this" : "static",
                sym->index);
    }
    
    advance(c); // ;
    advance(c);
}

void compileIf(Compiler *c) {
    int labelNum = c->labelCount++;
    
    advance(c); // (
    advance(c);
    compileExpression(c);
    advance(c); // )
    
    fprintf(c->output, "not\n");
    fprintf(c->output, "if-goto L%d\n", labelNum);
    
    advance(c); // {
    advance(c);
    compileStatements(c);
    advance(c); // }
    
    fprintf(c->output, "goto L%d\n", labelNum + 1);
    fprintf(c->output, "label L%d\n", labelNum);
    
    advance(c);
    if (strcmp(c->token, "else") == 0) {
        advance(c); // {
        advance(c);
        compileStatements(c);
        advance(c); // }
        advance(c);
    }
    
    fprintf(c->output, "label L%d\n", labelNum + 1);
    c->labelCount++;
}

void compileWhile(Compiler *c) {
    int labelNum = c->labelCount;
    c->labelCount += 2;
    
    fprintf(c->output, "label L%d\n", labelNum);
    
    advance(c); // (
    advance(c);
    compileExpression(c);
    advance(c); // )
    
    fprintf(c->output, "not\n");
    fprintf(c->output, "if-goto L%d\n", labelNum + 1);
    
    advance(c); // {
    advance(c);
    compileStatements(c);
    advance(c); // }
    
    fprintf(c->output, "goto L%d\n", labelNum);
    fprintf(c->output, "label L%d\n", labelNum + 1);
    
    advance(c);
}

void compileDo(Compiler *c) {
    advance(c);
    char name[MAX_TOKEN];
    strcpy(name, c->token);
    
    advance(c);
    int nArgs = 0;
    
    if (strcmp(c->token, ".") == 0) {
        advance(c);
        char fullName[MAX_TOKEN * 2];
        sprintf(fullName, "%s.%s", name, c->token);
        strcpy(name, fullName);
        advance(c); // (
        advance(c);
        nArgs = compileExpressionList(c);
    } else {
        advance(c);
        nArgs = compileExpressionList(c);
    }
    
    fprintf(c->output, "call %s %d\n", name, nArgs);
    fprintf(c->output, "pop temp 0\n");
    
    advance(c); // )
    advance(c); // ;
    advance(c);
}

void compileReturn(Compiler *c) {
    advance(c);
    if (strcmp(c->token, ";") != 0) {
        compileExpression(c);
    } else {
        fprintf(c->output, "push constant 0\n");
    }
    fprintf(c->output, "return\n");
    advance(c); // ;
    advance(c);
}

void compileTerm(Compiler *c) {
    if (c->tokenType == TK_INT_CONST) {
        fprintf(c->output, "push constant %s\n", c->token);
        advance(c);
    } else if (c->tokenType == TK_STRING_CONST) {
        int len = strlen(c->token);
        fprintf(c->output, "push constant %d\n", len);
        fprintf(c->output, "call String.new 1\n");
        for (int i = 0; i < len; i++) {
            fprintf(c->output, "push constant %d\n", c->token[i]);
            fprintf(c->output, "call String.appendChar 2\n");
        }
        advance(c);
    } else if (strcmp(c->token, "true") == 0) {
        fprintf(c->output, "push constant 1\n");
        fprintf(c->output, "neg\n");
        advance(c);
    } else if (strcmp(c->token, "false") == 0 || strcmp(c->token, "null") == 0) {
        fprintf(c->output, "push constant 0\n");
        advance(c);
    } else if (strcmp(c->token, "this") == 0) {
        fprintf(c->output, "push pointer 0\n");
        advance(c);
    } else if (strcmp(c->token, "(") == 0) {
        advance(c);
        compileExpression(c);
        advance(c); // )
    } else if (strcmp(c->token, "-") == 0 || strcmp(c->token, "~") == 0) {
        char op = c->token[0];
        advance(c);
        compileTerm(c);
        fprintf(c->output, "%s\n", op == '-' ? "neg" : "not");
    } else {
        char name[MAX_TOKEN];
        strcpy(name, c->token);
        advance(c);
        
        if (strcmp(c->token, "[") == 0) {
            Symbol *sym = findSymbol(c, name);
            advance(c);
            compileExpression(c);
            fprintf(c->output, "push %s %d\n", 
                    strcmp(sym->kind, "var") == 0 ? "local" : "argument",
                    sym->index);
            fprintf(c->output, "add\n");
            fprintf(c->output, "pop pointer 1\n");
            fprintf(c->output, "push that 0\n");
            advance(c); // ]
        } else if (strcmp(c->token, "(") == 0 || strcmp(c->token, ".") == 0) {
            if (strcmp(c->token, ".") == 0) {
                advance(c);
                char fullName[MAX_TOKEN * 2];
                sprintf(fullName, "%s.%s", name, c->token);
                strcpy(name, fullName);
                advance(c); // (
            }
            advance(c);
            int nArgs = compileExpressionList(c);
            fprintf(c->output, "call %s %d\n", name, nArgs);
            advance(c); // )
        } else {
            Symbol *sym = findSymbol(c, name);
            if (sym) {
                fprintf(c->output, "push %s %d\n", 
                        strcmp(sym->kind, "var") == 0 ? "local" : 
                        strcmp(sym->kind, "arg") == 0 ? "argument" :
                        strcmp(sym->kind, "field") == 0 ? "this" : "static",
                        sym->index);
            }
        }
    }
}

void compileExpression(Compiler *c) {
    compileTerm(c);
    
    while (strchr("+-*/&|<>=", c->token[0]) && strlen(c->token) == 1) {
        char op = c->token[0];
        advance(c);
        compileTerm(c);
        
        switch (op) {
            case '+': fprintf(c->output, "add\n"); break;
            case '-': fprintf(c->output, "sub\n"); break;
            case '*': fprintf(c->output, "call Math.multiply 2\n"); break;
            case '/': fprintf(c->output, "call Math.divide 2\n"); break;
            case '&': fprintf(c->output, "and\n"); break;
            case '|': fprintf(c->output, "or\n"); break;
            case '<': fprintf(c->output, "lt\n"); break;
            case '>': fprintf(c->output, "gt\n"); break;
            case '=': fprintf(c->output, "eq\n"); break;
        }
    }
}

int compileExpressionList(Compiler *c) {
    int count = 0;
    if (strcmp(c->token, ")") != 0) {
        compileExpression(c);
        count = 1;
        while (strcmp(c->token, ",") == 0) {
            advance(c);
            compileExpression(c);
            count++;
        }
    }
    return count;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s input.jack output.vm\n", argv[0]);
        return 1;
    }
    
    FILE *input = fopen(argv[1], "r");
    FILE *output = fopen(argv[2], "w");
    
    if (!input || !output) {
        printf("Error opening files\n");
        return 1;
    }
    
    Compiler compiler;
    initCompiler(&compiler, input, output);
    compileClass(&compiler);
    
    fclose(input);
    fclose(output);
    
    printf("Compilation completed: %s -> %s\n", argv[1], argv[2]);
    return 0;
}