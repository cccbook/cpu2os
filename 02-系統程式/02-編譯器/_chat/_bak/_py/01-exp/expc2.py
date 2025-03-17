import re

class Lexer:
    def __init__(self, expression):
        self.tokens = re.findall(r'\d+|[+\-*/()]', expression)
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
        else:
            token = self.current_token
            self.eat()
            return int(token)
    
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

expression = "3 + 5 * 2"
lexer = Lexer(expression)
parser = Parser(lexer)
print(parser.expr())  # 輸出 16