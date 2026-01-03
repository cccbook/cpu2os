# 1. 定義 Z Combinator (適用於 Python 的 Y Combinator)
# Z = λf. (λx. f(λv. x(x)(v))) (λx. f(λv. x(x)(v)))
Z = lambda f: (lambda x: f(lambda v: x(x)(v))) (lambda x: f(lambda v: x(x)(v)))

# 2. 定義階乘邏輯 G (Step Function)
# 這裡的 'self' 參數就是由 Z Combinator 傳入的遞迴功能
G = lambda self: lambda n: 1 if n == 0 else n * self(n - 1)

# 3. 組合出階乘函數
factorial = Z(G)

# 4. 測試
print(factorial(5)) # 輸出 120