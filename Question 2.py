# 4 / 1 / 2022
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


h = 0.1
H = int(1 / h)
u = np.zeros((H + 1, 2*H + 1))
M = 2*H
N = H

for i in range(2*H+1):
    u[0, i] = ((i*h)-1)**4
    u[H, i] = 1-6*((i*h)-1)**2+((i*h)-1)**4

for j in range(H+1):
    u[j, 0] = u[j, 2*H] = 1-6*(j*h)**2+(j*h)**4


U = [[] for _ in range(N-1)]
for n in range(0, N-1):
    U[n] = np.array([0 for _ in range(0, M)], dtype=np.float64)
U[0] = np.array([u[0, i] for i in range(M-1)], dtype=np.float64)
U[N-2] = np.array([u[N, i] for i in range(M-1)], dtype=np.float64)

Uini1 = [[] for _ in range(N-1)]
for n in range(N-1):
    Uini1[n] = np.array([0 for _ in range(M-1)], dtype=np.float64)
    Uini1[n][0] = u[n, 0]
    Uini1[n][M-2] = u[n, M]

Uini2 = [[] for _ in range(N-1)]
for n in range(N-1):
    Uini2[n] = np.array([0 for _ in range(M-1)], dtype=np.float64)
Uini2[0] = U[0]
Uini2[N-2] = U[N-2]

upper_diag = np.ones((M-1, M-1), dtype=np.float64)
center_diag = np.ones((M-1, M-1), dtype=np.float64)*(-4)
lower_diag = np.ones((M-1, M-1), dtype=np.float64)
B = np.eye(M-1)*center_diag
np.fill_diagonal(B[:-1, 1:], upper_diag)
np.fill_diagonal(B[1:, :-1], lower_diag)


A = np.zeros(((N-1)*(M-1),(N-1)*(M-1)), dtype=np.float64)
for m in range(0, (N-1)*(M-1), M-1):
    for n in range(0, (N-1)*(M-1), M-1):
        if m == n:
            A[m:m+M-1, n:n+M-1] = B
        if m == n-M+1:
            A[m:m+M-1, n:n+M-1] = np.eye(19)
        if m == n+M+1:
            A[m:m+M-1, n:n+M-1] = np.eye(19)

Urini1 = np.zeros(((M-1)*(N-1),1))
for m in range(0, (N-1)*(M-1), M-1):
    Urini1[m:m+M-1, 0] = Uini1[int(m/(M+1))]

Urini2 = np.zeros(((M-1)*(N-1),1))
Urini2[0:M-1, 0] = Uini2[0]
Urini2[(M-1)*(N-2):(M-1)*(N-1), 0] = Uini2[N-2]
U = (-1) * np.dot(np.linalg.inv(A),(Urini1+Urini2))

# reshape
for j in range(N-1):
    for i in range(M-1):
        u[j+1, i+1] = U[j*(M-1)+i]

utrue = np.zeros((H + 1, 2*H + 1))
for j in range(N+1):
    for i in range(M+1):
        x = h*i-1
        y = h*j
        utrue[j, i] = x**4+y**4-6*x**2*y**2

r=np.abs(u-utrue)

fig = plt.figure()
sub = fig.add_subplot(111, projection='3d')
X = np.arange(-1, 1+0.1*h, h)
Y = np.arange(0, 1+0.1*h, h)
X_mesh, Y_mesh = np.meshgrid(X, Y)
Z_mesh = r
sub.plot_surface(X_mesh,Y_mesh,Z_mesh,rstride=1,cstride=1,cmap='rainbow')
plt.title('R')
plt.xlabel('x')
plt.ylabel('y')
sub.set_zlim(0,1.5)
plt.show()

data_df = pd.DataFrame(u)
data_df.to_excel(os.getcwd()+'\\R.xlsx', index=False)
os.startfile(os.getcwd()+'\\R.xlsx')


