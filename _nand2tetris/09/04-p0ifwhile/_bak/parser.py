import re
import traceback

class Tokenizer:
    def __init__(self, code: str):
        # 匹配規則：關鍵字 | 關係運算符 | 數學運算符/括號/分號/大括號 | 數字 | 變數
        token_specification = [
            ('KEYWORD', r'let|print|if|else|while'),
            ('RELOP', r'==|!=|<=|>=|<|>'),
            ('OP', r'[+\-*/()]'),
            ('ASSIGN', r'='),
            ('SEPARATOR', r'[;{}]'),
            ('NUM', r'\d+(\.\d*)?'),
            # 【關鍵修正點】：將 ID 的匹配從 [a-zA-Z] 改為 [a-zA-Z]+
            ('ID', r'[a-zA-Z]+'),  # 匹配一個或多個字母
        ]
        
        # 為了確保 'let' 等關鍵字優先於 'ID' 匹配，關鍵字必須放在 ID 之前
        tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in token_specification)
        
        # 忽略空格和換行
        self._all_tokens = [
            m for m in re.finditer(tok_regex, code.strip()) 
            if not re.match(r'\s', m.group(0))
        ]
        
        self.tokens = self._all_tokens
        self.current_index = 0
        
    # ... (get_next, peek, match, print_all_tokens 方法保持不變) ...
    # 為了程式碼完整性，以下提供 print_all_tokens
    def get_next(self):
        if self.current_index < len(self.tokens):
            token = self.tokens[self.current_index]
            self.current_index += 1
            return token.lastgroup, token.group(0)
        return None, None

    def peek(self):
        if self.current_index < len(self.tokens):
            token = self.tokens[self.current_index]
            return token.lastgroup, token.group(0)
        return None, None
    
    def match(self, expected_type, expected_value=None):
        tok_type, tok_val = self.get_next()
        if tok_type != expected_type or (expected_value and tok_val != expected_value):
            raise SyntaxError(f"Expected {expected_type} ('{expected_value}' if specified) at index {self.current_index - 1}, but got {tok_type} ('{tok_val}')")
        return tok_val

    def print_all_tokens(self):
        token_list = []
        for token_match in self._all_tokens:
            token_list.append({
                'type': token_match.lastgroup,
                'value': token_match.group(0)
            })
        
        print("\n--- Tokenizer 輸出所有 Token ---")
        for i, token in enumerate(token_list):
            print(f"[{i:02d}] {token['type']:<9}: {token['value']}")
        print("---------------------------------")
        return token_list

class Interpreter:
    def __init__(self, code: str):
        self.tokenizer = Tokenizer(code)
        self.variables = {}  # 儲存所有變數
        
    def _parse_and_run(self):
        """
        模擬 <program> ::= <statement> | <statement> <program>
        """

        while self.tokenizer.peek()[0] is not None:
            self._statement()

    def _block(self):
        """
        模擬 <block> ::= { <program> }
        """
        self.tokenizer.match('SEPARATOR', '{')
        
        # 遞迴呼叫 _parse_and_run 來執行區塊內的語句
        self._parse_and_run() 
        
        self.tokenizer.match('SEPARATOR', '}')

    def _statement(self):
        """
        模擬 <statement> ::= <assignment> | <print_stmt> | <if_stmt> | <while_stmt>
        """
        tok_type, tok_val = self.tokenizer.peek()
        
        if tok_val == 'let':
            self._assignment()
        elif tok_val == 'print':
            self._print_stmt()
        elif tok_val == 'if':
            self._if_stmt()
        elif tok_val == 'while':
            self._while_stmt()
        elif tok_val == '}':
            # 這是區塊結尾，讓上層的 _block 處理
            return
        else:
            raise SyntaxError(f"Unexpected start of statement: {tok_val}")

    # --- 語句實現 ---

    def _assignment(self):
        """
        模擬 <assignment> ::= let <id> = <expression> ;
        """
        self.tokenizer.match('KEYWORD', 'let')
        var_name = self.tokenizer.match('ID') # 取得變數名
        self.tokenizer.match('ASSIGN', '=')
        
        value = self._expression() # 計算表達式的值
        self.variables[var_name] = value
        
        self.tokenizer.match('SEPARATOR', ';')

    def _print_stmt(self):
        """
        模擬 <print_stmt> ::= print <expression> ;
        """
        self.tokenizer.match('KEYWORD', 'print')
        value = self._expression() # 計算表達式的值
        
        print(f"OUTPUT: {value}")
        
        self.tokenizer.match('SEPARATOR', ';')

    def _if_stmt(self):
        """
        模擬 <if_stmt> ::= if ( <condition> ) <block> <else_clause>
        """
        self.tokenizer.match('KEYWORD', 'if')
        self.tokenizer.match('OP', '(')
        condition_result = self._condition()
        self.tokenizer.match('OP', ')')
        
        # 儲存當前 tokenizer 狀態，以便在 else 區塊跳過 if 區塊的語法
        start_index = self.tokenizer.current_index 
        
        if condition_result:
            self._block() # 執行 if 區塊
            
            # 處理 <else_clause> ::= else <block> | \epsilon 
            # 如果 if 條件為真，我們必須跳過 else 區塊（如果存在）
            if self.tokenizer.peek()[1] == 'else':
                self.tokenizer.get_next() # 消耗 'else'
                self._skip_block() # 跳過 else 區塊的語法
        else:
            # 跳過 if 區塊的語法
            self._skip_block()
            
            # 檢查 else 區塊是否存在
            if self.tokenizer.peek()[1] == 'else':
                self.tokenizer.match('KEYWORD', 'else')
                self._block() # 執行 else 區塊

    def _while_stmt(self):
        """
        模擬 <while_stmt> ::= while ( <condition> ) <block>
        """
        self.tokenizer.match('KEYWORD', 'while')
        
        # 儲存 while 條件和區塊的起始位置，用於迴圈回跳
        condition_start = self.tokenizer.current_index 
        
        # 外部迴圈：不斷檢查條件
        while True:
            # 每次迴圈開始前，重設到條件判斷的位置
            self.tokenizer.current_index = condition_start 
            
            self.tokenizer.match('OP', '(')
            condition_result = self._condition()
            self.tokenizer.match('OP', ')')
            
            block_start = self.tokenizer.current_index
            
            if condition_result:
                # 執行區塊
                self._block() 
                # 設置下次迴圈開始檢查條件
                self.tokenizer.current_index = condition_start 
            else:
                # 條件不滿足，跳過區塊並退出迴圈
                self.tokenizer.current_index = block_start
                self._skip_block() 
                break

    def _skip_block(self):
        """
        在 if/else/while 中，用於跳過一個語句塊的語法，不執行代碼。
        """
        balance = 0
        while True:
            tok_type, tok_val = self.tokenizer.get_next()
            if tok_val == '{':
                balance += 1
            elif tok_val == '}':
                balance -= 1
                if balance < 0:
                    break
            elif tok_type is None:
                raise SyntaxError("Unexpected end of code while skipping block.")

    # --- 條件與表達式實現 ---

    def _condition(self):
        """
        模擬 <condition> ::= <expression> <relop> <expression>
        """
        left = self._expression()
        
        relop = self.tokenizer.match('RELOP')
        
        right = self._expression()
        
        # 執行關係運算
        if relop == '==':
            return left == right
        elif relop == '!=':
            return left != right
        elif relop == '<':
            return left < right
        elif relop == '>':
            return left > right
        elif relop == '<=':
            return left <= right
        elif relop == '>=':
            return left >= right
        
        return False

    def _expression(self):
        """
        模擬 <expression> ::= <expression> <op> <expression> | ( <expression> ) | <id> | <num>
        (與之前一樣，從左到右計算，忽略標準優先級)
        """
        result = self._factor() 

        # 循環處理 E op E 結構
        while True:
            op_type, op_val = self.tokenizer.peek()
            
            if op_val in ['+', '-', '*', '/']:
                self.tokenizer.get_next() # 消耗運算符
                right = self._factor()
                
                # 執行數學運算 (從左到右)
                if op_val == '+':
                    result += right
                elif op_val == '-':
                    result -= right
                elif op_val == '*':
                    result *= right
                elif op_val == '/':
                    if right == 0:
                        raise ZeroDivisionError("Division by zero")
                    result /= right
            else:
                break
        return result

    def _factor(self):
        """
        處理最小單元：(E) | id | num
        """
        tok_type, tok_val = self.tokenizer.peek()

        if tok_val == '(':
            self.tokenizer.match('OP', '(')
            result = self._expression()
            self.tokenizer.match('OP', ')')
            return result
        
        elif tok_type == 'NUM':
            self.tokenizer.get_next()
            return float(tok_val)
        
        elif tok_type == 'ID':
            self.tokenizer.get_next()
            if tok_val not in self.variables:
                raise NameError(f"Variable '{tok_val}' not defined.")
            return self.variables[tok_val]
        
        else:
            raise SyntaxError(f"Unexpected token in expression: {tok_val}")

    def run(self):
        """運行整個程式，並在錯誤時輸出 Call Stack。"""
        # 運行前先打印所有 Token
        self.tokenizer.print_all_tokens()

        print("--- 極簡程式語言解釋器啟動 ---")
        try:
            self._parse_and_run()
            print("--- 執行完成 ---")
        except Exception as e:
            # 捕獲所有異常，並輸出堆疊追蹤資訊
            print("\n--- 執行錯誤 (帶 Call Stack) ---")
            
            # 輸出錯誤類型和訊息
            print(f"錯誤: {type(e).__name__}: {e}")
            
            # 使用 traceback.print_exc() 輸出完整的堆疊追蹤
            traceback.print_exc()
            
            print("---------------------------------")

# --- 範例程式碼 ---

sample_code = """
let i = 1;
let sum = 0;

while (i <= 5) {
    let sum = sum + i;
    let i = i + 1;
    
    if (i == 3) {
        print sum;
    } else {
        print i;
    }
}
print sum;
"""

# 執行範例程式
interpreter = Interpreter(sample_code)
try:
    interpreter.run()
except Exception as e:
    print(f"\n執行錯誤: {e}")