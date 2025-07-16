# 高階語言的語法 (BNF/EBNF)

## 簡易運算式語法

* https://github.com/cccbook/sp/tree/master/code/c/02-compiler/01-compileExp

```
E=T ([+-] T)*
T=F ([*/] F)*
F=Number | Id | '(' E ')'
```

## 詞彙語法描述

* [正規表達式](regexp)

```
String = ".*"
Number = [0-9]+
Id     = [A-Z_][A-Z0-9_]*
Char   = .
```

## 包含 while 的語法

```
PROG    = STMT
STMT    = WHILE | BLOCK | ASSIGN
WHILE   = while (E) STMT
BLOCK   = { STMT* }
ASSIGN  = id '=' E;
E       = F (op E)*
F       = (E) | Number | Id
```

## C0 語言

```
PROG     = (INCLUDE | DECL | FUNCTION)*
INCLUDE  = #...>
DECL     = *? ASSIGN
FUNCTION = type id (PARAM_LIST) BLOCK
STMT     = WHILE | IF | BLOCK | RETURN | VAR ; | ASSIGN;
WHILE    = while (E) STMT
IF       = if (E) STMT (else STMT)?
BLOCK    = { LIST<VAR> STMT* }
RETURN   = return E;
ASSIGN   = P (=E)?
CALL(id) = id ( LIST<E> )
VAR      = Type LIST<DECL>
E        = F (op F)*
F        = (++ | -- | [*&+-~!])? P
P        = ((E) | Number | Literal | ID) ([E] | (LIST<E>) | ++ | --)      // ID (LIST<E>)   == CALL(id)
```

## Lua 的語法

```

	chunk ::= {stat [`;´]} [laststat [`;´]]

	block ::= chunk

	stat ::=  varlist `=´ explist | 
		 functioncall | 
		 do block end | 
		 while exp do block end | 
		 repeat block until exp | 
		 if exp then block {elseif exp then block} [else block] end | 
		 for Name `=´ exp `,´ exp [`,´ exp] do block end | 
		 for namelist in explist do block end | 
		 function funcname funcbody | 
		 local function Name funcbody | 
		 local namelist [`=´ explist] 

	laststat ::= return [explist] | break

	funcname ::= Name {`.´ Name} [`:´ Name]

	varlist ::= var {`,´ var}

	var ::=  Name | prefixexp `[´ exp `]´ | prefixexp `.´ Name 

	namelist ::= Name {`,´ Name}

	explist ::= {exp `,´} exp

	exp ::=  nil | false | true | Number | String | `...´ | function | 
		 prefixexp | tableconstructor | exp binop exp | unop exp 

	prefixexp ::= var | functioncall | `(´ exp `)´

	functioncall ::=  prefixexp args | prefixexp `:´ Name args 

	args ::=  `(´ [explist] `)´ | tableconstructor | String 

	function ::= function funcbody

	funcbody ::= `(´ [parlist] `)´ block end

	parlist ::= namelist [`,´ `...´] | `...´

	tableconstructor ::= `{´ [fieldlist] `}´

	fieldlist ::= field {fieldsep field} [fieldsep]

	field ::= `[´ exp `]´ `=´ exp | Name `=´ exp | exp

	fieldsep ::= `,´ | `;´

	binop ::= `+´ | `-´ | `*´ | `/´ | `^´ | `%´ | `..´ | 
		 `<´ | `<=´ | `>´ | `>=´ | `==´ | `~=´ | 
		 and | or

	unop ::= `-´ | not | `#´
```

## C 語言的 BNF 語法

* http://sp1.wikidot.com/csyntax

## Jack 的語法 -- nand2tetris

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
