import sys
import re

# ==========================================
# 1. AST Nodes (抽象語法樹節點) - 無變更
# ==========================================
class Node:
    pass

class Number(Node):
    def __init__(self, value): self.value = value
    def __repr__(self): return f"Num({self.value})"

class Variable(Node):
    def __init__(self, name): self.name = name
    def __repr__(self): return f"Var({self.name})"

class BinaryOp(Node):
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right
    def __repr__(self): return f"Bin({self.left} {self.op} {self.right})"

class Lambda(Node):
    def __init__(self, param, body):
        self.param = param
        self.body = body
    def __repr__(self): return f"λ{self.param}.{self.body}"

class Call(Node):
    def __init__(self, func, arg):
        self.func = func
        self.arg = arg
    def __repr__(self): return f"Call({self.func}, {self.arg})"

class IfElse(Node):
    def __init__(self, condition, true_branch, false_branch):
        self.condition = condition
        self.true_branch = true_branch
        self.false_branch = false_branch
    def __repr__(self): return f"If({self.condition}, {self.true_branch}, {self.false_branch})"

class Assign(Node):
    def __init__(self, name, value):
        self.name = name
        self.value = value
    def __repr__(self): return f"Assign({self.name} = {self.value})"

class Print(Node):
    def __init__(self, expression): self.expression = expression

# ==========================================
# 2. Lexer (詞法分析器) - **已修改**
# ==========================================
class Lexer:
    def __init__(self, text):
        self.text = text
        self.pos = 0
        self.tokens = self.tokenize()

    def tokenize(self):
        token_spec = [
            ('NUMBER',   r'\d+'),
            ('IF',       r'if'),
            ('ELSE',     r'else'),
            ('LAMBDA',   r'lambda'),
            ('PRINT',    r'print'),
            ('ID',       r'[a-zA-Z_][a-zA-Z0-9_]*'),
            ('EQ',       r'=='),
            ('ASSIGN',   r'='),
            ('OP',       r'[+*\-]'),
            ('LPAREN',   r'\('),
            ('RPAREN',   r'\)'),
            ('COLON',    r':'),
            ('COMMENT',  r'#.*'),      # <--- 新增：匹配 # 開頭直到行尾
            ('SKIP',     r'[ \t\n]+'), # 跳過空白與換行
            ('MISMATCH', r'.'),
        ]
        tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in token_spec)
        tokens = []
        for mo in re.finditer(tok_regex, self.text):
            kind = mo.lastgroup
            value = mo.group()
            
            # 如果是 SKIP 或 COMMENT，直接忽略，不加入 Token 列表
            if kind in ('SKIP', 'COMMENT'):
                continue
            elif kind == 'MISMATCH':
                raise RuntimeError(f'Unexpected character: {value!r}')
            
            tokens.append((kind, value))
        tokens.append(('EOF', None))
        return tokens

# ==========================================
# 3. Parser (語法分析器) - 無變更
# ==========================================
class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def current_token(self):
        return self.tokens[self.pos]

    def eat(self, kind):
        if self.current_token()[0] == kind:
            self.pos += 1
        else:
            raise RuntimeError(f"Expected {kind}, got {self.current_token()[0]}")

    def parse(self):
        statements = []
        while self.current_token()[0] != 'EOF':
            statements.append(self.parse_statement())
        return statements

    def parse_statement(self):
        token = self.current_token()
        if token[0] == 'PRINT':
            self.eat('PRINT')
            self.eat('LPAREN')
            expr = self.parse_expression()
            self.eat('RPAREN')
            return Print(expr)
        elif token[0] == 'ID':
            if self.tokens[self.pos + 1][0] == 'ASSIGN':
                name = token[1]
                self.eat('ID')
                self.eat('ASSIGN')
                value = self.parse_expression()
                return Assign(name, value)
        return self.parse_expression()

    def parse_expression(self):
        if self.current_token()[0] == 'LAMBDA':
            return self.parse_lambda()
        return self.parse_ternary()

    def parse_lambda(self):
        self.eat('LAMBDA')
        param = self.current_token()[1]
        self.eat('ID')
        self.eat('COLON')
        body = self.parse_expression()
        return Lambda(param, body)

    def parse_ternary(self):
        node = self.parse_comparison()
        if self.current_token()[0] == 'IF':
            self.eat('IF')
            condition = self.parse_expression()
            self.eat('ELSE')
            else_node = self.parse_expression()
            return IfElse(condition, node, else_node)
        return node

    def parse_comparison(self):
        node = self.parse_term()
        while self.current_token()[0] == 'EQ':
            op = self.current_token()[1]
            self.eat('EQ')
            right = self.parse_term()
            node = BinaryOp(node, op, right)
        return node

    def parse_term(self):
        node = self.parse_call()
        while self.current_token()[0] == 'OP':
            op = self.current_token()[1]
            self.eat('OP')
            right = self.parse_call()
            node = BinaryOp(node, op, right)
        return node

    def parse_call(self):
        node = self.parse_atom()
        while self.current_token()[0] == 'LPAREN':
            self.eat('LPAREN')
            arg = self.parse_expression()
            self.eat('RPAREN')
            node = Call(node, arg)
        return node

    def parse_atom(self):
        token = self.current_token()
        kind, value = token
        
        if kind == 'LPAREN':
            self.eat('LPAREN')
            node = self.parse_expression()
            self.eat('RPAREN')
            return node
        elif kind == 'NUMBER':
            self.eat('NUMBER')
            return Number(int(value))
        elif kind == 'ID':
            self.eat('ID')
            return Variable(value)
        else:
            raise RuntimeError(f"Unexpected token in atom: {kind}")

# ==========================================
# 4. Interpreter (解譯器) - 無變更
# ==========================================
class Environment:
    def __init__(self, parent=None):
        self.vars = {}
        self.parent = parent

    def get(self, name):
        if name in self.vars:
            return self.vars[name]
        if self.parent:
            return self.parent.get(name)
        raise RuntimeError(f"Undefined variable: {name}")

    def set(self, name, value):
        self.vars[name] = value

class Closure:
    def __init__(self, param, body, env):
        self.param = param
        self.body = body
        self.env = env
    
    def __repr__(self):
        return f"<Closure λ{self.param}>"

class Interpreter:
    def __init__(self):
        self.global_env = Environment()

    def eval(self, node, env):
        if isinstance(node, Number):
            return node.value
        elif isinstance(node, Variable):
            return env.get(node.name)
        elif isinstance(node, BinaryOp):
            left = self.eval(node.left, env)
            right = self.eval(node.right, env)
            if node.op == '+': return left + right
            if node.op == '-': return left - right
            if node.op == '*': return left * right
            if node.op == '==': return left == right
        elif isinstance(node, IfElse):
            cond = self.eval(node.condition, env)
            if cond:
                return self.eval(node.true_branch, env)
            else:
                return self.eval(node.false_branch, env)
        elif isinstance(node, Lambda):
            return Closure(node.param, node.body, env)
        elif isinstance(node, Call):
            func = self.eval(node.func, env)
            arg = self.eval(node.arg, env)
            if not isinstance(func, Closure):
                raise RuntimeError(f"Attempting to call a non-function: {func}")
            new_env = Environment(func.env)
            new_env.set(func.param, arg)
            return self.eval(func.body, new_env)
        elif isinstance(node, Assign):
            val = self.eval(node.value, env)
            env.set(node.name, val)
        elif isinstance(node, Print):
            val = self.eval(node.expression, env)
            print(val)

    def run(self, code):
        lexer = Lexer(code)
        # 如果需要除錯，可以取消下面這行的註解來看看 tokens
        # print(lexer.tokens) 
        parser = Parser(lexer.tokens)
        statements = parser.parse()
        for stmt in statements:
            self.eval(stmt, self.global_env)

# ==========================================
# 5. Main Entry Point
# ==========================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python lambda_interpreter_v2.py <source_file>")
        sys.exit(1)

    file_path = sys.argv[1]

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
            
        interpreter = Interpreter()
        interpreter.run(source_code)
        
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error occurred: {e}")
        sys.exit(1)