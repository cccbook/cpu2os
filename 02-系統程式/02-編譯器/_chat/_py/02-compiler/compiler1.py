from enum import Enum
import re

# Token 類型定義
class TokenType(Enum):
    NUMBER = 'NUMBER'
    PLUS = 'PLUS'
    MINUS = 'MINUS'
    MULTIPLY = 'MULTIPLY'
    DIVIDE = 'DIVIDE'
    LPAREN = 'LPAREN'
    RPAREN = 'RPAREN'
    PRINT = 'PRINT'
    STRING = 'STRING'
    COMMA = 'COMMA'
    EQUALS = 'EQUALS'

class Token:
    def __init__(self, type, value):
        self.type = type
        self.value = value

class Lexer:
    def __init__(self, text):
        self.text = text
        self.pos = 0
        self.current_char = self.text[self.pos] if text else None

    def advance(self):
        self.pos += 1
        self.current_char = self.text[self.pos] if self.pos < len(self.text) else None

    def skip_whitespace(self):
        while self.current_char and self.current_char.isspace():
            self.advance()

    def get_number(self):
        result = ''
        while self.current_char and (self.current_char.isdigit() or self.current_char == '.'):
            result += self.current_char
            self.advance()
        return float(result)

    def get_string(self):
        self.advance()  # Skip first quote
        result = ''
        while self.current_char and self.current_char != '"':
            result += self.current_char
            self.advance()
        self.advance()  # Skip closing quote
        return result

    def get_tokens(self):
        tokens = []
        while self.current_char:
            if self.current_char.isspace():
                self.skip_whitespace()
                continue
            
            if self.current_char.isdigit():
                tokens.append(Token(TokenType.NUMBER, self.get_number()))
                continue
                
            if self.current_char == '"':
                tokens.append(Token(TokenType.STRING, self.get_string()))
                continue

            if self.current_char == '+':
                tokens.append(Token(TokenType.PLUS, '+'))
                self.advance()
                continue

            if self.current_char == '-':
                tokens.append(Token(TokenType.MINUS, '-'))
                self.advance()
                continue

            if self.current_char == '*':
                tokens.append(Token(TokenType.MULTIPLY, '*'))
                self.advance()
                continue

            if self.current_char == '/':
                tokens.append(Token(TokenType.DIVIDE, '/'))
                self.advance()
                continue

            if self.current_char == '(':
                tokens.append(Token(TokenType.LPAREN, '('))
                self.advance()
                continue

            if self.current_char == ')':
                tokens.append(Token(TokenType.RPAREN, ')'))
                self.advance()
                continue

            if self.current_char == ',':
                tokens.append(Token(TokenType.COMMA, ','))
                self.advance()
                continue

            if self.current_char == '=':
                tokens.append(Token(TokenType.EQUALS, '='))
                self.advance()
                continue

            if self.current_char.isalpha():
                # 處理關鍵字
                word = ''
                while self.current_char and self.current_char.isalpha():
                    word += self.current_char
                    self.advance()
                if word == 'print':
                    tokens.append(Token(TokenType.PRINT, word))
                continue

            raise Exception(f'Invalid character: {self.current_char}')

        return tokens

class ASTNode:
    pass

class BinOpNode(ASTNode):
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right

class NumberNode(ASTNode):
    def __init__(self, value):
        self.value = value

class StringNode(ASTNode):
    def __init__(self, value):
        self.value = value

class PrintNode(ASTNode):
    def __init__(self, args):
        self.args = args

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.current_token = tokens[0] if tokens else None

    def advance(self):
        self.pos += 1
        self.current_token = self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def factor(self):
        token = self.current_token
        if token.type == TokenType.NUMBER:
            self.advance()
            return NumberNode(token.value)
        elif token.type == TokenType.STRING:
            self.advance()
            return StringNode(token.value)
        elif token.type == TokenType.LPAREN:
            self.advance()
            result = self.expr()
            if self.current_token.type != TokenType.RPAREN:
                raise Exception("Missing closing parenthesis")
            self.advance()
            return result
        raise Exception("Invalid factor")

    def term(self):
        result = self.factor()
        while self.current_token and self.current_token.type in (TokenType.MULTIPLY, TokenType.DIVIDE):
            op = self.current_token
            self.advance()
            result = BinOpNode(result, op, self.factor())
        return result

    def expr(self):
        result = self.term()
        while self.current_token and self.current_token.type in (TokenType.PLUS, TokenType.MINUS):
            op = self.current_token
            self.advance()
            result = BinOpNode(result, op, self.term())
        return result

    def print_statement(self):
        self.advance()  # Skip 'print'
        if self.current_token.type != TokenType.LPAREN:
            raise Exception("Expected '(' after print")
        self.advance()
        
        args = []
        while True:
            args.append(self.expr())
            if self.current_token.type == TokenType.RPAREN:
                break
            if self.current_token.type != TokenType.COMMA:
                raise Exception("Expected ',' or ')'")
            self.advance()
        
        self.advance()  # Skip ')'
        return PrintNode(args)

    def parse(self):
        if self.current_token.type == TokenType.PRINT:
            return self.print_statement()
        return self.expr()

class Compiler:
    def __init__(self):
        self.instructions = []
        self.temp_counter = 0

    def visit(self, node):
        if isinstance(node, NumberNode):
            return str(node.value)
        elif isinstance(node, StringNode):
            return f'"{node.value}"'
        elif isinstance(node, BinOpNode):
            left = self.visit(node.left)
            right = self.visit(node.right)
            temp = f't{self.temp_counter}'
            self.temp_counter += 1
            self.instructions.append(f'{temp} = {left} {node.op.value} {right}')
            return temp
        elif isinstance(node, PrintNode):
            args = []
            for arg in node.args:
                result = self.visit(arg)
                args.append(result)
            self.instructions.append(f'PRINT {", ".join(args)}')
            return None

    def compile(self, ast):
        self.visit(ast)
        return self.instructions

def run_compiler(source_code):
    # 詞法分析
    lexer = Lexer(source_code)
    tokens = lexer.get_tokens()
    
    # 語法分析
    parser = Parser(tokens)
    ast = parser.parse()
    
    # 生成中間碼
    compiler = Compiler()
    intermediate_code = compiler.compile(ast)
    
    return intermediate_code

# 測試編譯器
source_code = 'print("3+(2*5)=", 3+(2*5))'
intermediate_code = run_compiler(source_code)

print("Source code:")
print(source_code)
print("\nIntermediate code:")
for instruction in intermediate_code:
    print(instruction)