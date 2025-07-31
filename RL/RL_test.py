import numpy as np
import random

P = np.random.normal(1, 0.1, 10)
print(f'P:\n{P}')
lamda = 0.99
A = [0.01] * 10
print(f'A:\n{A}')

def update_p(P, lamda, A):
    # Pupdate = (P + lamda * A) - min(P + lamda * A) 不能直接广播相乘，用列表推导式
    Pupdate = (P + [lamda * a for a in A]) - min(P + [lamda * a for a in A])
    print(f'Pupdate:\n{Pupdate}')

if __name__ == '__main__':
    n = 2
    for _ in range(n):
        update_p(P, lamda, A)
