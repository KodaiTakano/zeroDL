import numpy as np

def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x) # xと同じ配列形状
    
    for idx in range(x.size):
        tmp_val = x[idx] # 後で元に戻すときに使う
        # f(x + h)
        x[idx] = tmp_val + h # x[0]=x[0]+h, x[1]=x[1]+h
        fxh1 = f(x)
        # f(x - h)
        x[idx] = tmp_val - h
        fxh2 = f(x)
        
        grad[idx] = (fxh1 - fxh2) / (2 * h) # まずx0の偏微分, 2周目でx1の偏微分
        x[idx] = tmp_val # 元に戻す
        
    return grad

def function_2(x):
    return x[0]**2 + x[1]**2

print(numerical_gradient(function_2, np.array([3.0, 4.0])))
print(numerical_gradient(function_2, np.array([0.0, 2.0])))
print(numerical_gradient(function_2, np.array([3.0, 0.0])))