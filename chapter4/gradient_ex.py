import numpy as np

# 数値微分
def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2 * h)

def function_1(x):
    return 0.01 * x**2 + 0.1 * x

print(numerical_diff(function_1, 5)) # 0.2
print(numerical_diff(function_1, 10)) # 0.3
