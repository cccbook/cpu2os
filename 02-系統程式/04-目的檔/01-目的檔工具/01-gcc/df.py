def df(f, x, h=0.001):
    return (f(x+h)-f(x))/h

print('df(x**2, 3)=', df(lambda x:x**2, 3))
