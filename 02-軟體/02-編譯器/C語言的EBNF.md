# C 語言的 EBNF

```ebnf
%token int_const char_const float_const id string enumeration_const

translation_unit	: external_decl+

external_decl		: decl_spec* declarator decl* compound_stat
				/* function definition */
			| decl

decl_spec		: storage_class_spec | type_spec | type_qualifier

decl			: decl_spec+ init_declarator_list? ';'

storage_class_spec	: 'auto' | 'register' | 'static' | 'extern' | 'typedef'

type_spec		: 'void' | 'char' | 'short' | 'int' | 'long' | 'float'
			| 'double' | 'signed' | 'unsigned'
			| struct_or_union id? '{' struct_decl+ '}'
			| struct_or_union id
			| 'enum' id? '{' enumerator ( ',' enumerator )* '}'
			| 'enum' id
			| typedef_name

type_qualifier		: 'const' | 'volatile'

struct_or_union		: 'struct' | 'union'

init_declarator_list	: init_declarator ( ',' init_declarator )*

init_declarator		: declarator
			| declarator '=' initializer

struct_decl		: spec_qualifier_list struct_declarator
				( ',' struct_declarator )* ';'

spec_qualifier_list	: ( type_spec | type_qualifier )+

struct_declarator	: declarator
			| declarator? ':' const_exp

enumerator		: id
			| id '=' const_exp

declarator		: pointer? direct_declarator

direct_declarator	: id
			| '(' declarator ')'
			| direct_declarator '[' const_exp? ']'
			| direct_declarator '(' param_type_list ')'
			| direct_declarator '(' id_list? ')'

pointer			: ( '*' type_qualifier* )+

param_type_list		: param_decl ( ',' param_decl )* ( ',' '...' | )

param_decl		: decl_spec+ ( declarator | abstract_declarator? )

id_list			: id ( ',' id )*

initializer		: assignment_exp
			| '{' initializer ( ',' initializer )* ( ',' | ) '}'

type_name		: spec_qualifier_list abstract_declarator?

abstract_declarator	: pointer
			| pointer? direct_abstract_declarator

direct_abstract_declarator: '(' abstract_declarator ')'
				( '[' const_exp? ']'
				| '(' param_type_list? ')' )*

typedef_name		: id

stat			: id ':' stat
			| 'case' const_exp ':' stat
			| 'default' ':' stat
			| exp? ';'
			| compound_stat
			| 'if' '(' exp ')' stat
			| 'if' '(' exp ')' stat 'else' stat
			| 'switch' '(' exp ')' stat
			| 'while' '(' exp ')' stat
			| 'do' stat 'while' '(' exp ')' ';'
			| 'for' '(' exp? ';' exp? ';' exp? ')' stat
			| 'goto' id ';'
			| 'continue' ';'
			| 'break' ';'
			| 'return' exp? ';'

compound_stat		: '{' decl* stat* '}'

exp			: assignement_exp ( ',' assignment_exp )*

assignment_exp		: ( unary_exp assignment_operator )* conditional_exp

assignment_operator	: '=' | '*=' | '/=' | '%=' | '+=' | '-=' | '<<='
			| '>>=' | '&=' | '^=' | '|='

conditional_exp		: ( logical_or_exp '?' exp ':' )* logical_or_exp

const_exp		: conditional_exp

%left '*' '/' '%'
%left '+' '-'
%left '<<' '>>'
%left '<' '>' '<=' '>='
%left '==' '!='
%left '&'
%left '^'
%left '|'
%left '&&'
%left '||'

logical_or_exp		: cast_exp
				( ( '*' | '/' | '%' | '+' | '-' | '<<' | '>>'
				  | '<' | '>' | '<=' | '>=' | '==' | '!='
				  | '&' | '^' | '|' | '&&' | '||' )
				 cast_exp )*

cast_exp		: ( '(' type_name ')' )* unary_exp

unary_exp		: ( id | int_const | char_const | float_const
				| enumeration_const | string | '(' exp ')'
			  )
			  (	'[' exp ']'
				| '(' argument_exp_list? ')'
				| '.' id
				| '->' id
				| '++' | '--'			%left!
			  )*
			| '++' unary_exp
			| '--' unary_exp
			| unary_operator cast_exp
			| 'sizeof' unary_exp
			| 'sizeof' '(' type_name ')'

unary_operator		: '&' | '*' | '+' | '-' | '~' | '!'

argument_exp_list	: assignment_exp ( ',' assignment_exp )*
```