import sys
import re
import os

# 設定遞迴深度以容納 Church Numerals 的深層運算
sys.setrecursionlimit(50000)

# ==========================================
# 1. 詞法分析 (Lexer)
# ==========================================

class Token:
    def __init__(self, type, value):
        self.type = type
        self.value = value
    def __repr__(self):
        return f"Token({self.type}, {repr(self.value)})"

class Lexer:
    def __init__(self, text):
        # 預處理：處理 Python 的行接續符號 '\'
        self.text = text.replace('\\\n', '').replace('\\\r\n', '')
        self.pos = 0
        self.current_char = self.text[0] if self.text else None
        self.tokens = []

    def error(self, msg):
        raise Exception(f"Lexer Error: {msg}")

    def advance(self):
        self.pos += 1
        if self.pos < len(self.text):
            self.current_char = self.text[self.pos]
        else:
            self.current_char = None

    def peek_char(self):
        """偷看下一個字元，不移動指標"""
        peek_pos = self.pos + 1
        if peek_pos < len(self.text):
            return self.text[peek_pos]
        return None

    def skip_whitespace(self):
        while self.current_char is not None and self.current_char.isspace():
            self.advance()

    def skip_comment(self):
        while self.current_char is not None and self.current_char != '\n':
            self.advance()
        self.advance() # Skip newline

    def identifier(self):
        result = ''
        while self.current_char is not None and (self.current_char.isalnum() or self.current_char == '_'):
            result += self.current_char
            self.advance()
        
        if result == 'lambda':
            return Token('LAMBDA', 'lambda')
        # [修正] 移除 print 的特殊判斷，讓它被視為普通的 ID
        return Token('ID', result)

    def string_literal(self):
        # 處理 'string' 或 f'string'
        is_fstring = False
        if self.current_char == 'f':
            is_fstring = True
            self.advance() # skip 'f'
        
        quote_type = self.current_char
        self.advance() # skip opening quote
        
        result = ''
        while self.current_char is not None and self.current_char != quote_type:
            # 簡單處理 escape char
            if self.current_char == '\\':
                self.advance()
                if self.current_char == 'x': # hex escape (例如顏色碼)
                    self.advance()
                    if self.pos + 1 < len(self.text):
                        hex_val = self.text[self.pos:self.pos+2]
                        try:
                            result += chr(int(hex_val, 16))
                        except ValueError:
                            result += '\\x' + hex_val 
                        self.pos += 1 # skip 2nd hex digit
                        self.advance()
                    else:
                        result += '\\x'
                elif self.current_char == 'n':
                    result += '\n'
                    self.advance()
                elif self.current_char == 't':
                    result += '\t'
                    self.advance()
                elif self.current_char in ["'", '"', '\\']:
                    result += self.current_char
                    self.advance()
                else:
                    result += '\\' + self.current_char 
                    self.advance()
            else:
                result += self.current_char
                self.advance()
        
        if self.current_char == quote_type:
            self.advance() # skip closing quote
            
        return Token('FSTRING' if is_fstring else 'STRING', result)

    def tokenize(self):
        while self.current_char is not None:
            if self.current_char.isspace():
                self.skip_whitespace()
                continue
            
            if self.current_char == '#':
                self.skip_comment()
                continue

            # 優先檢查是否為 f-string (f 開頭且後面接引號)
            if self.current_char == 'f' and self.peek_char() in ("'", '"'):
                self.tokens.append(self.string_literal())
                continue

            if self.current_char.isalpha() or self.current_char == '_':
                self.tokens.append(self.identifier())
                continue
            
            if self.current_char in ("'", '"'):
                self.tokens.append(self.string_literal())
                continue
            
            if self.current_char == '=':
                self.tokens.append(Token('ASSIGN', '='))
                self.advance()
                continue

            if self.current_char == ':':
                self.tokens.append(Token('COLON', ':'))
                self.advance()
                continue

            if self.current_char == '(':
                self.tokens.append(Token('LPAREN', '('))
                self.advance()
                continue

            if self.current_char == ')':
                self.tokens.append(Token('RPAREN', ')'))
                self.advance()
                continue

            self.error(f"Unexpected character: {self.current_char}")
        
        self.tokens.append(Token('EOF', None))
        return self.tokens

# ==========================================
# 2. 抽象語法樹節點 (AST Nodes)
# ==========================================

class AST: pass

class Var(AST):
    def __init__(self, name):
        self.name = name
    def __repr__(self): return f"Var({self.name})"

class Lambda(AST):
    def __init__(self, param, body):
        self.param = param
        self.body = body
    def __repr__(self): return f"λ{self.param}.{self.body}"

class Application(AST):
    def __init__(self, func, arg):
        self.func = func
        self.arg = arg
    def __repr__(self): return f"({self.func} {self.arg})"

class Assignment(AST):
    def __init__(self, name, value):
        self.name = name
        self.value = value

class StringLit(AST):
    def __init__(self, value, is_fstring=False):
        self.value = value
        self.is_fstring = is_fstring
    def __repr__(self): return f"'{self.value}'"

# [修正] 移除 PrintStat 節點，因為 print 現在是普通函數呼叫

# ==========================================
# 3. 語法分析 (Parser)
# ==========================================

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.current_token = self.tokens[0]

    def error(self, msg):
        raise Exception(f"Parser Error: {msg} at {self.current_token}")

    def eat(self, token_type):
        if self.current_token.type == token_type:
            self.pos += 1
            if self.pos < len(self.tokens):
                self.current_token = self.tokens[self.pos]
        else:
            self.error(f"Expected {token_type}, got {self.current_token.type}")

    def parse(self):
        statements = []
        while self.current_token.type != 'EOF':
            statements.append(self.statement())
        return statements

    def statement(self):
        # ID = ...
        if self.current_token.type == 'ID' and self.peek().type == 'ASSIGN':
            var_name = self.current_token.value
            self.eat('ID')
            self.eat('ASSIGN')
            expr = self.expression()
            return Assignment(var_name, expr)
        
        # [修正] 移除對 PRINT token 的特殊處理
        # 讓它自然進入 expression -> parse_application -> atom (ID 'print')

        expr = self.expression()
        return expr

    def peek(self):
        if self.pos + 1 < len(self.tokens):
            return self.tokens[self.pos + 1]
        return Token('EOF', None)

    def expression(self):
        if self.current_token.type == 'LAMBDA':
            return self.parse_lambda()
        return self.parse_application()

    def parse_lambda(self):
        self.eat('LAMBDA')
        param = self.current_token.value
        self.eat('ID')
        self.eat('COLON')
        body = self.expression()
        return Lambda(param, body)

    def parse_application(self):
        node = self.atom()
        while self.current_token.type == 'LPAREN':
            self.eat('LPAREN')
            arg = self.expression()
            self.eat('RPAREN')
            node = Application(node, arg)
        return node

    def atom(self):
        token = self.current_token
        
        if token.type == 'ID':
            self.eat('ID')
            return Var(token.value)
        
        elif token.type == 'STRING':
            self.eat('STRING')
            return StringLit(token.value, is_fstring=False)

        elif token.type == 'FSTRING':
            self.eat('FSTRING')
            return StringLit(token.value, is_fstring=True)

        elif token.type == 'LPAREN':
            self.eat('LPAREN')
            expr = self.expression()
            self.eat('RPAREN')
            return expr

        else:
            self.error(f"Unexpected token in atom: {token}")

# ==========================================
# 4. 解譯器核心 (Interpreter)
# ==========================================

class Environment:
    def __init__(self, parent=None):
        self.vars = {}
        self.parent = parent

    def define(self, name, value):
        self.vars[name] = value

    def lookup(self, name):
        if name in self.vars:
            return self.vars[name]
        if self.parent:
            return self.parent.lookup(name)
        raise Exception(f"Runtime Error: Variable '{name}' not found")

class Closure:
    def __init__(self, param, body, env):
        self.param = param
        self.body = body
        self.env = env 
    def __repr__(self):
        return f"<Closure λ{self.param}>"

# [修正] 新增 NativeFunction 類別來處理 print 等內建函數
class NativeFunction:
    def __init__(self, func, name="native"):
        self.func = func
        self.name = name
    def __repr__(self):
        return f"<NativeFunction {self.name}>"

class Interpreter:
    def __init__(self):
        self.global_env = Environment()
        # [修正] 在全域環境註冊 print 函數
        self.global_env.define('print', NativeFunction(lambda x: print(x), 'print'))

    def eval(self, node, env):
        if isinstance(node, Var):
            return env.lookup(node.name)
        
        elif isinstance(node, Lambda):
            return Closure(node.param, node.body, env)
        
        elif isinstance(node, Application):
            func = self.eval(node.func, env)
            arg_val = self.eval(node.arg, env)
            
            # [修正] 支援呼叫 NativeFunction
            if isinstance(func, NativeFunction):
                return func.func(arg_val)
            
            if not isinstance(func, Closure):
                raise Exception(f"Attempting to call non-function: {func}")

            new_env = Environment(func.env)
            new_env.define(func.param, arg_val)
            
            return self.eval(func.body, new_env)

        elif isinstance(node, StringLit):
            if not node.is_fstring:
                return node.value
            
            # F-String 模擬: {var} 替換
            result = node.value
            matches = re.findall(r'\{([a-zA-Z0-9_]+)\}', result)
            for var_name in matches:
                try:
                    val = env.lookup(var_name)
                    result = result.replace(f'{{{var_name}}}', str(val))
                except:
                    pass
            return result
        
        elif isinstance(node, Assignment):
            val = self.eval(node.value, env)
            env.define(node.name, val)
            return val

        return None

    def execute(self, statements):
        for stmt in statements:
            try:
                self.eval(stmt, self.global_env)
            except Exception as e:
                print(f"Error executing statement: {e}")
                import traceback
                traceback.print_exc()

# ==========================================
# Main
# ==========================================

def main():
    target_file = 'lambdaCalculus.py'
    if len(sys.argv) > 1:
        target_file = sys.argv[1]

    if not os.path.exists(target_file):
        print(f"File not found: {target_file}")
        return

    print(f"--- Pure Python Interpreter: {target_file} ---\n")

    with open(target_file, 'r', encoding='utf-8') as f:
        code = f.read()

    lexer = Lexer(code)
    tokens = lexer.tokenize()
    
    parser = Parser(tokens)
    ast_nodes = parser.parse()
    
    interpreter = Interpreter()
    interpreter.execute(ast_nodes)

if __name__ == '__main__':
    main()