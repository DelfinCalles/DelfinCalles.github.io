import numpy as np
import math

def CoefDF(k, xbar, x):
    n = len(x)
    A = np.zeros((n, n))
    B = np.zeros((n, 1))
    h = min(x[1:n] - x[0:n-1])
    h2 = min(abs(x - xbar))
    if h2 > 0:
        h = min(h, h2)
    p = n - k
    for i in range(n):
        for j in range(n):
            A[i, j] = (x[j] - xbar) ** i / math.factorial(i)
    B[k] = 1
    coef = np.linalg.solve(A, B)
    coef = coef*h**k
    
    return coef
