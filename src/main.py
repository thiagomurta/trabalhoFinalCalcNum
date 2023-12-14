import matplotlib.pyplot as plt
import numpy as np

def g(x, valorLambda):
    return ((np.pi**2) * np.sin(np.pi * x)) - valorLambda * np.exp(np.sin(np.pi*x))

def Jacobiana(x, u, valorLambda,  quantValores):
    n = len(x)
    j = np.zeros([n, n])
    j[0][0] = 1
    j[n-1][n-1] = 1
    h = x[1] - x[0]
    for i in range(1, n-1):
        j[i, i-1] = -1
        j[i, i+1] = -1
        j[i,i] = 2 - h**2 * valorLambda * np.exp(u[i])
    return j

def residuo(x, u, valorLambda, quantValores):
    n = len(x)
    F = np.zeros([n])
    F[0]    = 0
    F[n-1]  = 0
    h = x[1] - x[0]

    for i in range(1, n-1):
        F[i] = -u[i-1] + 2*u[i] - u[i+1] - h**2 * valorLambda * np.exp(u[i]) - h**2*g(x[i], valorLambda)
    return F

def solveSystem(x, u, valorLambda, tol, kmax, quantValores):
    k = 0
    F = residuo(x, u, valorLambda, quantValores)
    error = []
    fnorm = np.linalg.norm(F)
    error.append(fnorm)

    while( fnorm > tol and k < kmax):
        print("Iter = %d |F| = %8.8e" %(k,fnorm))
        j = Jacobiana(x,u,valorLambda, quantValores)
        du = np.linalg.solve(j, -F)
        u = u + du
        F = residuo(x, u, valorLambda, quantValores)
        k = k+1
        fnorm = np.linalg.norm(F)
        error.append(fnorm)
    return u, error, x


def bratu_problem(N,_lambda):
    x = np.linspace(0,1,N)
    n = len(x)
    u = np.zeros([n])
    u, erro, x1 = solveSystem(x, u, _lambda, 1.0E-07, 100, N)
    return u, erro, x1


N_valores = [10,50,100]

print("-----10-----")
u, erro, x = bratu_problem(N_valores[0], 2)
print("Erro: ", np.linalg.norm(erro))
plt.plot(x, u, color='blue')

print("-----50-----")
u1, erro1, x1 = bratu_problem(N_valores[1], 2)
print("Erro: ", np.linalg.norm(erro1))
plt.plot(x1, u1, color='green')

print("-----100-----")
u2, erro2, x2 = bratu_problem(N_valores[2], 2)
print("Erro: ", np.linalg.norm(erro2))
plt.plot(x2, u2, color='red')

plt.xlabel('x')
plt.ylabel('u(x)')
plt.gca().legend(('N=10','N=50','N=100'))
plt.show()