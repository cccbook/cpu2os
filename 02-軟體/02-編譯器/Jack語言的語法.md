# nand2tetris -- Jack 語言的語法 

## BNF 語法

```
class : 'class' id '{' classVarDec* subroutineDec* '}'

classVarDec: {'static'|'field'} type varList ';'

varList: varName (',' varName)*

varName: ID

subroutineDec: ('constructor'|'function'|'method') ('void'|type) ID '(' parameterList? ')' subroutineBody

parameterList: (parameter (',' parameter)*)?

parameter: type varName

subroutineBody: '{' varDec* statements '}'

varDec: 'var' type varList ';'

statements : statement*

statement: letStatement | ifStatement | whileStatement | doStatement | returnStatement

letStatement: 'let' varName ( arraySubscript )? '=' expression ';'

arraySubscript: '[' expression ']'

block = '{' statements '}'

ifStatement : 'if' '(' expression ')' block '('else' block)?

whileStatement: 'while' '(' expression ')' block

doStatement: 'do' call ';'

returnStatement: 'return' expression? ';'

expression: term (op term)* 

term: integerConstant | stringConstant | keywordConstant | varName arraySubscript?| call | '(' expression ')' | op1 term

call: subroutineName '(' expressionList ')' | (className | varName) '.' subroutineName '(' expressionList ')'

expressionList: (expression (',' expression)*)?

```

## 程式範例

```
class Main {

   function void main() {
       do Output.printInt(1 + (2 * 3));
       return;
   }

}
```
