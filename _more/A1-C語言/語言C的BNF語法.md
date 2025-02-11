# C 語言的 BNF 語法

```
primary_exp : id
     | const
     | string
     | '(' exp ')'
     ;
postfix_exp : primary_exp                                      基本算式
     | postfix_exp '[' exp ']'                                 陣列存取
     | postfix_exp '(' argument_exp_list ')'         函數呼叫
     | postfix_exp '(' ')'                 函數呼叫   
     | postfix_exp '.' id                                        結構成員
     | postfix_exp '->' id                                     結構成員
     | postfix_exp '++'                                        x++
     | postfix_exp '--'                                          x--
     ;
argument_exp_list : assignment_exp
     | argument_exp_list ',' assignment_exp
     ;
function_definition : decl_specs declarator decl_list compound_stat   
    
EXAMPLE 1 In the following:
extern int max(int a, int b)
{
return a > b ? a : b;
}
extern is the storage-class specifier and int is the type specifier; max(int a, int b) is the
function declarator; and
{ return a > b ? a : b; }
is the function body. The following similar definition uses the identifier-list form for the parameter
declarations:
```


資料來源：http://c.comsci.us/syntax/expression/ebnf.html


```
: <expression> :  ::= <sequence-expression>
: <sequence-expression> : ::= <assignment-expression> ( <sequence-operator> <assignment-expression> ) *
: <assignment-expression> : ::= ( <variable-reference> <assignment-operator> ) * <conditional-expression>
: <variable-reference> : ::= <unary-expression>
: <conditional-expression> : ::= <logical-or-expression> ( '?' <expression> ':' <logical-or-expression> ) *
: <logical-or-expression> : ::= <logical-and-expression> ( <logical-or-operator> <logical-and-expression> ) *
: <logical-and-expression> : ::= <bitwise-or-expression> ( <logical-and-operator> <bitwise-or-expression> ) *
: <bitwise-or-expression> : ::= <bitwise-xor-expression> ( <bitwise-or-operator> <bitwise-xor-expression> ) *
: <bitwise-xor-expression> : ::= <bitwise-and-expression> ( <bitwise-xor-operator> <bitwise-and-expression> ) *
: <bitwise-and-expression> : ::= <equality-expression> ( <bitwise-and-operator> <equality-expression> ) *
: <equality-expression> : ::= <relational-expression> ( <equality-operator> <relational-expression> ) *
: <relational-expression> : ::= <shift-expression> ( <relational-operator> <shift-expression> ) *
: <shift-expression> : ::= <additive-expression> ( <shift-operator> <additive-expression> ) *
: <additive-expression> : ::= <multiplicative-expression> ( <additive-operator> <multiplicative-expression> ) *
: <multiplicative-expression> : ::= <cast-expression> ( <multiplicative-operator> <cast-expression> ) *
: <cast-expression> : ::= ( '(' <type-name> ')' ) * <unary-expression>
: <unary-expression> : ::= <unary-operator> <cast-expression>|  ( <prefix-operator> ) * <postfix-expression> |  'sizeof' '(' <type> ')'
: <postfix-expression> : ::= <primary-expression> <postfix-phrase>
: <postfix-phrase> : ::= <postfix-operator>|  <member-selector>|  <element-selector>|  <argument-specifier>
: <member-selector> : ::= '.' <identifier>|  '->' <identifier>
: <element-selector> : ::= '[' <expression> ']'
: <argument-specifier> : ::= '(' <argument> ( ',' <argument> ')' ')'
: <argument> : ::= <expression>
: <primary-expression> : ::= <identifier>|  <literal-constant>|  '(' <expression> ')'
```

http://lists.canonical.org/pipermail/kragen-hacks/1999-October/000201.html

```
translation-unit: (function-definition | declaration)+

function-definition:
  declaration-specifiers? declarator declaration* block

declaration: declaration-specifiers init-declarator% ";"

declaration-specifiers:
  (storage-class-specifier | type-specifier | type-qualifier)+

storage-class-specifier:
  ("auto" | "register" | "static" | "extern" | "typedef")

type-specifier: ("void" | "char" | "short" | "int" | "long" | "float" |
  "double" | "signed" | "unsigned" | struct-or-union-specifier |
  enum-specifier | typedef-name)

type-qualifier: ("const" | "volatile")

struct-or-union-specifier:
  ("struct" | "union") (
    identifier? "{" struct-declaration+ "}" |
    identifier
  )

init-declarator: declarator ("=" initializer)?

struct-declaration:
  (type-specifier | type-qualifier)+ struct-declarator%

struct-declarator: declarator | declarator? ":" constant-expression

enum-specifier: "enum" (identifier | identifier? "{" enumerator% "}")

enumerator: identifier ("=" constant-expression)?

declarator:
  pointer? (identifier | "(" declarator ")") (
    "[" constant-expression? "]" |
    "(" parameter-type-list ")" |
    "(" identifier%? ")"
  )*

pointer:
  ("*" type-qualifier*)*

parameter-type-list: parameter-declaration% ("," "...")?

parameter-declaration:
  declaration-specifiers (declarator | abstract-declarator)?

initializer: assignment-expression | "{" initializer% ","? "}"

type-name: (type-specifier | type-qualifier)+ abstract-declarator?

abstract-declarator:
  pointer ("(" abstract-declarator ")")? (
    "[" constant-expression? "]" |
    "(" parameter-type-list? ")"
  )*

statement:
  ((identifier | "case" constant-expression | "default") ":")*
  (expression? ";" |
   block |
   "if" "(" expression ")" statement |
   "if" "(" expression ")" statement "else" statement |
   "switch" "(" expression ")" statement |
   "while" "(" expression ")" statement |
   "do" statement "while" "(" expression ")" ";" |
   "for" "(" expression? ";" expression? ";" expression? ")" statement |
   "goto" identifier ";" |
   "continue" ";" |
   "break" ";" |
   "return" expression? ";"
  )

block: "{" declaration* statement* "}"

expression:
  assignment-expression%

assignment-expression: (
    unary-expression (
      "=" | "*=" | "/=" | "%=" | "+=" | "-=" | "<<=" | ">>=" | "&=" |
      "^=" | "|="
    )
  )* conditional-expression

conditional-expression:
  logical-OR-expression ( "?" expression ":" conditional-expression )?

constant-expression: conditional-expression

logical-OR-expression:
  logical-AND-expression ( "||" logical-AND-expression )*

logical-AND-expression:
  inclusive-OR-expression ( "&&" inclusive-OR-expression )*

inclusive-OR-expression:
  exclusive-OR-expression ( "|" exclusive-OR-expression )*

exclusive-OR-expression:
  AND-expression ( "^" AND-expression )*

AND-expression:
  equality-expression ( "&" equality-expression )*

equality-expression:
  relational-expression ( ("==" | "!=") relational-expression )*

relational-expression:
  shift-expression ( ("<" | ">" | "<=" | ">=") shift-expression )*

shift-expression:
  additive-expression ( ("<<" | ">>") additive-expression )*

additive-expression:
  multiplicative-expression ( ("+" | "-") multiplicative-expression )*

multiplicative-expression:
  cast-expression ( ("*" | "/" | "%") cast-expression )*

cast-expression:
  ( "(" type-name ")" )* unary-expression
  
unary-expression:
  ("++" | "--" | "sizeof" ) * (
    "sizeof" "(" type-name ")"                           |
    ("&" | "*" | "+" | "-" | "~" | "!" ) cast-expression |
    postfix-expression
  )

postfix-expression:
  (identifier | constant | string | "(" expression ")") (
    "[" expression "]"             |
    "(" assignment-expression% ")" |
    "." identifier                 |
    "->" identifier                |
    "++"                           |
    "--"
  )*

constant:
  integer-constant |
  character-constant |
  floating-constant |
  enumeration-constant
```


http://shell.sh.cvut.cz/~wilx/repos/c2pas/_darcs/current/docs/c_syntax.bnf

```
%token int_const char_const float_const id string enumeration_const
%%

translation_unit : external_decl
     | translation_unit external_decl
     ;
external_decl : function_definition
     | decl
     ;
function_definition : decl_specs declarator decl_list compound_stat
     | declarator decl_list compound_stat
     | decl_specs declarator compound_stat
     | declarator compound_stat
     ;
decl : decl_specs init_declarator_list ';'
     | decl_specs ';'
     ;
decl_list : decl
     | decl_list decl
     ;
decl_specs : storage_class_spec decl_specs
     | storage_class_spec
     | type_spec decl_specs
     | type_spec
     | type_qualifier decl_specs
     | type_qualifier
     ;
storage_class_spec : 'auto' | 'register' | 'static' | 'extern' | 'typedef'
     ;
type_spec : 'void' | 'char' | 'short' | 'int' | 'long' | 'float'
     | 'double' | 'signed' | 'unsigned'
     | struct_or_union_spec
     | enum_spec
     | typedef_name
     ;
type_qualifier : 'const' | 'volatile'
     ;
struct_or_union_spec : struct_or_union id '{' struct_decl_list '}'
     | struct_or_union '{' struct_decl_list '}'
     | struct_or_union id
     ;
struct_or_union : 'struct' | 'union'
     ;
struct_decl_list : struct_decl
     | struct_decl_list struct_decl
     ;
init_declarator_list : init_declarator
     | init_declarator_list ',' init_declarator
     ;
init_declarator : declarator
     | declarator '=' initializer
     ;
struct_decl : spec_qualifier_list struct_declarator_list ';'
     ;
spec_qualifier_list : type_spec spec_qualifier_list
     | type_spec
     | type_qualifier spec_qualifier_list
     | type_qualifier
     ;
struct_declarator_list : struct_declarator
     | struct_declarator_list ',' struct_declarator
     ;
struct_declarator : declarator
     | declarator ':' const_exp
     | ':' const_exp
     ;
enum_spec : 'enum' id '{' enumerator_list '}'
     | 'enum' '{' enumerator_list '}'
     | 'enum' id
     ;
enumerator_list : enumerator
     | enumerator_list ',' enumerator
     ;
enumerator : id
     | id '=' const_exp
     ;
declarator : pointer direct_declarator
     | direct_declarator
     ;
direct_declarator : id
     | '(' declarator ')'
     | direct_declarator '[' const_exp ']'
     | direct_declarator '[' ']'
     | direct_declarator '(' param_type_list ')'
     | direct_declarator '(' id_list ')'
     | direct_declarator '(' ')'
     ;
pointer : '*' type_qualifier_list
     | '*'
     | '*' type_qualifier_list pointer
     | '*' pointer
     ;
type_qualifier_list : type_qualifier
     | type_qualifier_list type_qualifier
     ;
param_type_list : param_list
     | param_list ',' '...'
     ;
param_list : param_decl
     | param_list ',' param_decl
     ;
param_decl : decl_specs declarator
     | decl_specs abstract_declarator
     | decl_specs
     ;
id_list : id
     | id_list ',' id
     ;
initializer : assignment_exp
     | '{' initializer_list '}'
     | '{' initializer_list ',' '}'
     ;
initializer_list : initializer
     | initializer_list ',' initializer
     ;
type_name : spec_qualifier_list abstract_declarator
     | spec_qualifier_list
     ;
abstract_declarator : pointer
     | pointer direct_abstract_declarator
     | direct_abstract_declarator
     ;
direct_abstract_declarator: '(' abstract_declarator ')'
     | direct_abstract_declarator '[' const_exp ']'
     | '[' const_exp ']'
     | direct_abstract_declarator '[' ']'
     | '[' ']'
     | direct_abstract_declarator '(' param_type_list ')'
     | '(' param_type_list ')'
     | direct_abstract_declarator '(' ')'
     | '(' ')'
     ;
typedef_name : id
     ;
stat : labeled_stat
     | exp_stat
     | compound_stat
     | selection_stat
     | iteration_stat
     | jump_stat
     ;
labeled_stat : id ':' stat
     | 'case' const_exp ':' stat
     | 'default' ':' stat
     ;
exp_stat : exp ';'
     | ';'
     ;
compound_stat : '{' decl_list stat_list '}'
     | '{' stat_list '}'
     | '{' decl_list '}'
     | '{' '}'
     ;
stat_list : stat
     | stat_list stat
     ;
selection_stat : 'if' '(' exp ')' stat
     | 'if' '(' exp ')' stat 'else' stat
     | 'switch' '(' exp ')' stat
     ;
iteration_stat : 'while' '(' exp ')' stat
     | 'do' stat 'while' '(' exp ')' ';'
     | 'for' '(' exp ';' exp ';' exp ')' stat
     | 'for' '(' exp ';' exp ';' ')' stat
     | 'for' '(' exp ';' ';' exp ')' stat
     | 'for' '(' exp ';' ';' ')' stat
     | 'for' '(' ';' exp ';' exp ')' stat
     | 'for' '(' ';' exp ';' ')' stat
     | 'for' '(' ';' ';' exp ')' stat
     | 'for' '(' ';' ';' ')' stat
     ;
jump_stat : 'goto' id ';'
     | 'continue' ';'
     | 'break' ';'
     | 'return' exp ';'
     | 'return' ';'
     ;
exp : assignment_exp
     | exp ',' assignment_exp
     ;
assignment_exp : conditional_exp
     | unary_exp assignment_operator assignment_exp
     ;
assignment_operator : '=' | '*=' | '/=' | '%=' | '+=' | '-=' | '<<='
     | '>>=' | '&=' | '^=' | '|='
     ;
conditional_exp : logical_or_exp
     | logical_or_exp '?' exp ':' conditional_exp
     ;
const_exp : conditional_exp
     ;
logical_or_exp : logical_and_exp
     | logical_or_exp '||' logical_and_exp
     ;
logical_and_exp : inclusive_or_exp
     | logical_and_exp '&&' inclusive_or_exp
     ;
inclusive_or_exp : exclusive_or_exp
     | inclusive_or_exp '|' exclusive_or_exp
     ;
exclusive_or_exp : and_exp
     | exclusive_or_exp '^' and_exp
     ;
and_exp : equality_exp
     | and_exp '&' equality_exp
     ;
equality_exp : relational_exp
     | equality_exp '==' relational_exp
     | equality_exp '!=' relational_exp
     ;
relational_exp : shift_expression
     | relational_exp '<' shift_expression
     | relational_exp '>' shift_expression
     | relational_exp '<=' shift_expression
     | relational_exp '>=' shift_expression
     ;
shift_expression : additive_exp
     | shift_expression '<<' additive_exp
     | shift_expression '>>' additive_exp
     ;
additive_exp : mult_exp
     | additive_exp '+' mult_exp
     | additive_exp '-' mult_exp
     ;
mult_exp : cast_exp
     | mult_exp '*' cast_exp
     | mult_exp '/' cast_exp
     | mult_exp '%' cast_exp
     ;
cast_exp : unary_exp
     | '(' type_name ')' cast_exp
     ;
unary_exp : postfix_exp
     | '++' unary_exp
     | '--' unary_exp
     | unary_operator cast_exp
     | 'sizeof' unary_exp
     | 'sizeof' '(' type_name ')'
     ;
unary_operator : '&' | '*' | '+' | '-' | '~' | '!'
     ;
postfix_exp : primary_exp
     | postfix_exp '[' exp ']'
     | postfix_exp '(' argument_exp_list ')'
     | postfix_exp '(' ')'
     | postfix_exp '.' id
     | postfix_exp '->' id
     | postfix_exp '++'
     | postfix_exp '--'
     ;
primary_exp : id
     | const
     | string
     | '(' exp ')'
     ;
argument_exp_list : assignment_exp
     | argument_exp_list ',' assignment_exp
     ;
const : int_const
     | char_const
     | float_const
     | enumeration_const
     ;
```