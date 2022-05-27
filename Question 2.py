# 4 / 1 / 2022
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def fun1(x):
    h = x
    H = int(1 / h)
    u = np.zeros((H + 1, 2 * H + 1))
    M = 2 * H
    N = H

    for i in range(2 * H + 1):
        u[0, i] = ((i * h) - 1) ** 4
        u[H, i] = 1 - 6 * ((i * h) - 1) ** 2 + ((i * h) - 1) ** 4

    for j in range(H + 1):
        u[j, 0] = u[j, 2 * H] = 1 - 6 * (j * h) ** 2 + (j * h) ** 4

    U = [[] for _ in range(N - 1)]
    for n in range(0, N - 1):
        U[n] = np.array([0 for _ in range(0, M)], dtype=np.float64)
    U[0] = np.array([u[0, i] for i in range(M - 1)], dtype=np.float64)
    U[N - 2] = np.array([u[N, i] for i in range(M - 1)], dtype=np.float64)

    Uini1 = [[] for _ in range(N - 1)]
    for n in range(N - 1):
        Uini1[n] = np.array([0 for _ in range(M - 1)], dtype=np.float64)
        Uini1[n][0] = u[n, 0]
        Uini1[n][M - 2] = u[n, M]

    Uini2 = [[] for _ in range(N - 1)]
    for n in range(N - 1):
        Uini2[n] = np.array([0 for _ in range(M - 1)], dtype=np.float64)
    Uini2[0] = U[0]
    Uini2[N - 2] = U[N - 2]

    upper_diag = np.ones((M - 1, M - 1), dtype=np.float64)
    center_diag = np.ones((M - 1, M - 1), dtype=np.float64) * (-4)
    lower_diag = np.ones((M - 1, M - 1), dtype=np.float64)
    B = np.eye(M - 1) * center_diag
    np.fill_diagonal(B[:-1, 1:], upper_diag)
    np.fill_diagonal(B[1:, :-1], lower_diag)

    A = np.zeros(((N - 1) * (M - 1), (N - 1) * (M - 1)), dtype=np.float64)
    for m in range(0, (N - 1) * (M - 1), M - 1):
        for n in range(0, (N - 1) * (M - 1), M - 1):
            if m == n:
                A[m:m + M - 1, n:n + M - 1] = B
            if m == n - M + 1:
                A[m:m + M - 1, n:n + M - 1] = np.eye(M-1)
            if m == n + M + 1:
                A[m:m + M - 1, n:n + M - 1] = np.eye(M-1)

    Urini1 = np.zeros(((M - 1) * (N - 1), 1))
    for m in range(0, (N - 1) * (M - 1), M - 1):
        Urini1[m:m + M - 1, 0] = Uini1[int(m / (M + 1))]

    Urini2 = np.zeros(((M - 1) * (N - 1), 1))
    Urini2[0:M - 1, 0] = Uini2[0]
    Urini2[(M - 1) * (N - 2):(M - 1) * (N - 1), 0] = Uini2[N - 2]
    U = (-1) * np.dot(np.linalg.inv(A), (Urini1 + Urini2))

    # reshape
    for j in range(N - 1):
        for i in range(M - 1):
            u[j + 1, i + 1] = U[j * (M - 1) + i]
    return u


def fun2(x):
    h = x
    H = int(1 / h)
    u = np.zeros((H + 1, 2 * H + 1))
    M = 2 * H
    N = H

    for i in range(2 * H + 1):
        u[0, i] = ((i * h) - 1) ** 4
        u[H, i] = 1 - 6 * ((i * h) - 1) ** 2 + ((i * h) - 1) ** 4

    for j in range(H + 1):
        u[j, 0] = u[j, 2 * H] = 1 - 6 * (j * h) ** 2 + (j * h) ** 4

    U = [[] for _ in range(N - 1)]
    for n in range(0, N - 1):
        U[n] = np.array([0 for _ in range(0, M)], dtype=np.float64)
    U[0] = np.array([u[0, i] for i in range(M - 1)], dtype=np.float64)
    U[N - 2] = np.array([u[N, i] for i in range(M - 1)], dtype=np.float64)

    Uini1 = [[] for _ in range(N - 1)]
    for n in range(N - 1):
        Uini1[n] = np.array([0 for _ in range(M - 1)], dtype=np.float64)
        Uini1[n][0] = u[n, 0]
        Uini1[n][M - 2] = u[n, M]

    Uini2 = [[] for _ in range(N - 1)]
    for n in range(N - 1):
        Uini2[n] = np.array([0 for _ in range(M - 1)], dtype=np.float64)
    Uini2[0] = U[0]
    Uini2[N - 2] = U[N - 2]

    upper_diag = np.ones((M - 1, M - 1), dtype=np.float64)
    center_diag = np.ones((M - 1, M - 1), dtype=np.float64) * (-4)
    lower_diag = np.ones((M - 1, M - 1), dtype=np.float64)
    B = np.eye(M - 1) * center_diag
    np.fill_diagonal(B[:-1, 1:], upper_diag)
    np.fill_diagonal(B[1:, :-1], lower_diag)

    A = np.zeros(((N - 1) * (M - 1), (N - 1) * (M - 1)), dtype=np.float64)
    for m in range(0, (N - 1) * (M - 1), M - 1):
        for n in range(0, (N - 1) * (M - 1), M - 1):
            if m == n:
                A[m:m + M - 1, n:n + M - 1] = B
            if m == n - M + 1:
                A[m:m + M - 1, n:n + M - 1] = np.eye(M-1)
            if m == n + M + 1:
                A[m:m + M - 1, n:n + M - 1] = np.eye(M-1)

    Urini1 = np.zeros(((M - 1) * (N - 1), 1))
    for m in range(0, (N - 1) * (M - 1), M - 1):
        Urini1[m:m + M - 1, 0] = Uini1[int(m / (M + 1))]

    Urini2 = np.zeros(((M - 1) * (N - 1), 1))
    Urini2[0:M - 1, 0] = Uini2[0]
    Urini2[(M - 1) * (N - 2):(M - 1) * (N - 1), 0] = Uini2[N - 2]
    U = (-1) * np.dot(np.linalg.inv(A), (Urini1 + Urini2))

    # reshape
    for j in range(N - 1):
        for i in range(M - 1):
            u[j + 1, i + 1] = U[j * (M - 1) + i]
    u = u.T
    return u

# def fun2(x):
#     h = x
#     H = int(1 / h)
#     u = np.zeros((H + 1, H + 1))
#     N = H
#
#     for i in range(H + 1):
#         u[0, i] = (i * h) ** 4
#         u[H, i] = 1 - 6 * (i * h) ** 2 + (i * h) ** 4
#
#     for j in range(H + 1):
#         u[j, 0] = (j * h)**4
#         u[j, H] = 1 - 6 * (j * h) ** 2 + (j * h) ** 4
#
#     U = [[] for _ in range(N - 1)]
#     for n in range(0, N - 1):
#         U[n] = np.array([0 for _ in range(0, N)], dtype=np.float64)
#     U[0] = np.array([u[0, i] for i in range(N - 1)], dtype=np.float64)
#     U[N - 2] = np.array([u[N, i] for i in range(N - 1)], dtype=np.float64)
#
#     Uini1 = [[] for _ in range(N - 1)]
#     for n in range(N - 1):
#         Uini1[n] = np.array([0 for _ in range(N - 1)], dtype=np.float64)
#         Uini1[n][0] = u[n, 0]
#         Uini1[n][N - 2] = u[n, N]
#
#     Uini2 = [[] for _ in range(N - 1)]
#     for n in range(N - 1):
#         Uini2[n] = np.array([0 for _ in range(N - 1)], dtype=np.float64)
#     Uini2[0] = U[0]
#     Uini2[N - 2] = U[N - 2]
#
#     upper_diag = np.ones((N - 1, N - 1), dtype=np.float64)
#     center_diag = np.ones((N - 1, N - 1), dtype=np.float64) * (-4)
#     lower_diag = np.ones((N - 1, N - 1), dtype=np.float64)
#     B = np.eye(N - 1) * center_diag
#     np.fill_diagonal(B[:-1, 1:], upper_diag)
#     np.fill_diagonal(B[1:, :-1], lower_diag)
#
#     A = np.zeros(((N - 1) * (N - 1), (N - 1) * (N - 1)), dtype=np.float64)
#     for m in range(0, (N - 1) * (N - 1), N - 1):
#         for n in range(0, (N - 1) * (N - 1), N - 1):
#             if m == n:
#                 A[m:m + N - 1, n:n + N - 1] = B
#             if m == n - N + 1:
#                 A[m:m + N - 1, n:n + N - 1] = np.eye(N-1)
#             if m == n + N + 1:
#                 A[m:m + N - 1, n:n + N - 1] = np.eye(N-1)
#
#     Urini1 = np.zeros(((N - 1) * (N - 1), 1))
#     for m in range(0, (N - 1) * (N - 1), N - 1):
#         Urini1[m:m + N - 1, 0] = Uini1[int(m / (N + 1))]
#
#     Urini2 = np.zeros(((N - 1) * (N - 1), 1))
#     Urini2[0:N - 1, 0] = Uini2[0]
#     Urini2[(N - 1) * (N - 2):(N - 1) * (N - 1), 0] = Uini2[N - 2]
#     U = (-1) * np.dot(np.linalg.inv(A), (Urini1 + Urini2))
#
#     # reshape
#
#     for j in range(N - 1):
#         for i in range(N - 1):
#             u[j + 1, i + 1] = U[j * (N - 1) + i]
#     np.flip(u, axis=0)
#     return u
#
#     # for j in range(N + 1):
#     #     for i in range(N + 1):
#     #         x = h * i - 1
#     #         y = h * j
#     #         u0[j, i] = 0
#     # u = np.hstack((u0, u))


def utrue1(x):
    h = x
    H = int(1 / h)
    M = 2 * H
    N = H
    utrue = np.zeros((H + 1, 2 * H + 1))
    for j in range(N + 1):
        for i in range(M + 1):
            x = h * i - 1
            y = h * j
            utrue[j, i] = x ** 4 + y ** 4 - 6 * x ** 2 * y ** 2
    return utrue


def utrue2(x):
    h = x
    H = int(1 / h)
    M = 2 * H
    N = H
    utrue = np.zeros((H + 1, 2 * H + 1))
    for j in range(N + 1):
        for i in range(M + 1):
            x = h * i - 1
            y = h * j
            utrue[j, i] = x ** 4 + y ** 4 - 6 * x ** 2 * y ** 2
    np.flip(utrue, axis=0)
    utrue = utrue.T
    return utrue


h = 0.1
utrue1 = utrue1(h)
utrue2 = utrue2(h)
u1 = fun1(h)
u2 = fun2(h)
r1 = np.abs(u1 - utrue1)
r2 = np.abs(u2 - utrue2)

# 原图
fig1 = plt.figure()

sub1 = fig1.add_subplot(111, projection='3d')
X1 = np.arange(-1, 1 + 0.1 * h, h)
Y1 = np.arange(0, 1 + 0.1 * h, h)
X1_mesh, Y1_mesh = np.meshgrid(X1, Y1)
Z1_mesh = u1
sub1.plot_surface(X1_mesh, Y1_mesh, Z1_mesh, rstride=1, cstride=1, cmap='rainbow')
plt.title('z')
plt.xlabel('x')
plt.ylabel('y')
sub1.set_zlim(-4, 4)

X2 = np.arange(0, 1 + 0.1 * h, h)
Y2 = np.arange(1, -1 - 0.1 * h, -h)
X2_mesh, Y2_mesh = np.meshgrid(X2, Y2)
Z2_mesh = u2
sub1.plot_surface(X2_mesh, Y2_mesh, Z2_mesh, rstride=1, cstride=1, cmap='rainbow')

# 误差图
fig2 = plt.figure()

sub2 = fig2.add_subplot(111, projection='3d')
X3 = np.arange(-1, 1 + 0.1 * h, h)
Y3 = np.arange(0, 1 + 0.1 * h, h)
X3_mesh, Y3_mesh = np.meshgrid(X1, Y1)
Z3_mesh = r1
sub2.plot_surface(X3_mesh, Y3_mesh, Z3_mesh, rstride=1, cstride=1, cmap='rainbow')
plt.title('r')
plt.xlabel('x')
plt.ylabel('y')
sub2.set_zlim(-1, 4)

X4 = np.arange(0, 1 + 0.1 * h, h)
Y4 = np.arange(1, -1 - 0.1 * h, -h)
X4_mesh, Y4_mesh = np.meshgrid(X4, Y4)
Z4_mesh = r2
sub2.plot_surface(X4_mesh, Y4_mesh, Z4_mesh, rstride=1, cstride=1, cmap='rainbow')

plt.show()

# data_df = pd.DataFrame(u)
# data_df.to_excel(os.getcwd() + '\\R.xlsx', index=False)
# os.startfile(os.getcwd() + '\\R.xlsx')
