class Interpreter:
    def __init__(self):
        # 存儲臨時變數的值
        self.variables = {}
    
    def evaluate_expression(self, expr):
        # 如果是數字字串，轉換為浮點數
        if isinstance(expr, (int, float)):
            return expr
        
        # 如果是字串常量（包含引號），返回字串內容
        if expr.startswith('"') and expr.endswith('"'):
            return expr[1:-1]
            
        # 如果是臨時變數，返回其值
        if expr in self.variables:
            return self.variables[expr]
            
        # 如果是數字字串，轉換為浮點數
        try:
            return float(expr)
        except ValueError:
            raise Exception(f"Unknown expression: {expr}")
    
    def execute_arithmetic(self, target, left, operator, right):
        # 計算左右運算元的值
        left_value = self.evaluate_expression(left)
        right_value = self.evaluate_expression(right)
        
        # 執行運算
        if operator == '+':
            result = left_value + right_value
        elif operator == '-':
            result = left_value - right_value
        elif operator == '*':
            result = left_value * right_value
        elif operator == '/':
            if right_value == 0:
                raise Exception("Division by zero")
            result = left_value / right_value
        else:
            raise Exception(f"Unknown operator: {operator}")
            
        # 儲存結果到臨時變數
        self.variables[target] = result
        return result
    
    def execute_print(self, args):
        # 計算所有參數的值
        evaluated_args = []
        for arg in args:
            arg = arg.strip()  # 移除可能的空白
            value = self.evaluate_expression(arg)
            evaluated_args.append(value)
        
        # 輸出結果
        print(*evaluated_args)
    
    def execute_line(self, line):
        # 忽略空行
        if not line.strip():
            return
            
        # 解析 PRINT 指令
        if line.startswith('PRINT'):
            # 提取 PRINT 後的參數
            args = line[5:].strip().split(',')
            self.execute_print(args)
            return
            
        # 解析賦值指令
        if '=' in line:
            # 分割等號左右兩邊
            target, expression = line.split('=', 1)
            target = target.strip()
            expression = expression.strip()
            
            # 分割運算元和運算符
            parts = expression.split()
            if len(parts) != 3:
                raise Exception(f"Invalid arithmetic expression: {expression}")
                
            left, operator, right = parts
            self.execute_arithmetic(target, left, operator, right)
    
    def execute(self, intermediate_code):
        # 依次執行每一行中間碼
        for line in intermediate_code:
            self.execute_line(line)
            
        # 返回所有臨時變數的值（用於調試）
        return self.variables

# 測試解譯器
def test_interpreter():
    # 測試用的中間碼
    intermediate_code = [
        't0 = 2 * 5',
        't1 = 3 + t0',
        'PRINT "3+(2*5)=", t1'
    ]
    
    # 建立解譯器實例
    interpreter = Interpreter()
    
    print("Executing intermediate code:")
    print("---------------------------")
    for line in intermediate_code:
        print(f"Executing: {line}")
        interpreter.execute_line(line)
    
    print("\nFinal variable values:")
    print("---------------------")
    for var, value in interpreter.variables.items():
        print(f"{var} = {value}")

if __name__ == "__main__":
    test_interpreter()