import re

def evaluate_function(name, args):
    if name == "print":
        print(*args)
    else:
        raise ValueError(f"未知的函式: {name}")

class Lexer:
    def __init__(self, expression):
        self.tokens = re.findall(r'[a-zA-Z_][a-zA-Z_0-9]*|\d+|[+\-*/(),]', expression)
        self.pos = 0
    
    def get_next_token(self):
        if self.pos < len(self.tokens):
            token = self.tokens[self.pos]
            self.pos += 1
            return token
        return None

class Parser:
    def __init__(self, lexer):
        self.lexer = lexer
        self.current_token = self.lexer.get_next_token()
    
    def eat(self):
        self.current_token = self.lexer.get_next_token()
    
    def factor(self):
        if self.current_token == '(':
            self.eat()
            result = self.expr()
            self.eat()  # 吃掉 ')'
            return result
        elif self.current_token.isdigit():
            token = self.current_token
            self.eat()
            return int(token)
        elif self.current_token.isalpha():
            return self.function_call()
    
    def function_call(self):
        func_name = self.current_token
        self.eat()
        self.eat()  # 吃掉 '('
        args = []
        while self.current_token != ')':
            args.append(self.expr())
            if self.current_token == ',':
                self.eat()
        self.eat()  # 吃掉 ')'
        return evaluate_function(func_name, args)
    
    def term(self):
        result = self.factor()
        while self.current_token in ('*', '/'):
            op = self.current_token
            self.eat()
            if op == '*':
                result *= self.factor()
            elif op == '/':
                result /= self.factor()
        return result
    
    def expr(self):
        result = self.term()
        while self.current_token in ('+', '-'):
            op = self.current_token
            self.eat()
            if op == '+':
                result += self.term()
            elif op == '-':
                result -= self.term()
        return result

expression = "print('3+(2*5)=', 3+(2*5))"
lexer = Lexer(expression)
parser = Parser(lexer)
parser.expr()  # 輸出 3+(2*5)= 13