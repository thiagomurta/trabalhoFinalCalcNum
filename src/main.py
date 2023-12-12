import matplotlib.pyplot as plt
import numpy as np

def g(x):
    return ((np.pi**2) * np.sin(np.pi * x)) - valorLambda * np.exp(np.sin(np.pi*x))

def Jacobiana(x, quantValores):
    j = np.zeros([quantValores, quantValores])
    
    for i in range(0, quantValores-1):
        j[i, i+1] = -1
        j[i+1, i] = -1
    
    for i in range(0, quantValores-1):
        j[i,i] = 2 - (h**2 * valorLambda * np.exp(x[0,i]))
    
    return j

def residuo(x, quantValores):
    F = np.zeros([quantValores,1])
    F[0,0] = x[0,0]
    F[quantValores-1,0] = x[0, quantValores-1]
    for i in range(1, quantValores-1):
        F[i,0] = (-x[0, i-1] + 2*x[0,i] - x[0,i+1])- h * valorLambda * np.exp(x[0,1]) - g(x[0, i])
    return F

def solveSystem(x, tol, kmax, quantValores):
    k = 0
    F = residuo(x, quantValores)
    while(np.linalg.norm(F) > tol and k < kmax):
        j = Jacobiana(x, quantValores)
        dx = np.linalg.solve(j, -F)
        x = x + dx
        F = residuo(x, quantValores)
        k = k+1
        erro = np.linalg.norm(dx)/np.linalg.norm(x)
    return k, F, erro


tol = 10e-7

N_valores = [10,50,100]
a = 0.
b = 1.

x = np.array([ np.linspace(a, b, N_valores[0]) ])
x1 = np.array([ np.linspace(a, b, N_valores[1]) ])
x2 = np.array([ np.linspace(a, b, N_valores[2]) ])

h = (b-a)/N_valores[0]

kmax = 100
tolerance = 10e-7
valorLambda = 2

k, F, erro = solveSystem(x, tol, kmax, N_valores[0])
k1, F1, erro1 = solveSystem(x1, tol, kmax, N_valores[1])
k2, F2, erro2 = solveSystem(x2, tol, kmax, N_valores[2])

y = np.linalg.norm(F)
y1 = np.linalg.norm(F1)
y2 = np.linalg.norm(F2)

print("------- 10 subdivisões -------")
print("Número de Interações: ", k)
print("Valor de ||F|| encontrado: ", y)
print("Erro: ", erro)
print()

print("------- 50 subdivisões -------")
print("Número de Interações: ", k1)
print("Valor de ||F|| encontrado: ", y1)
print("Erro: ", erro1)
print()

print("------- 100 subdivisões -------")
print("Número de Interações: ", k2)
print("Valor de ||F|| encontrado: ", y2)
print("Erro: ", erro2)

xk = np.linspace(0, k, 100)
xk1 = np.linspace(0, k1, 100)
xk2 = np.linspace(0, k2, 100)

yf = np.linspace(0, y, 100)
yf1 = np.linspace(0, y1, 100)
yf2 = np.linspace(0, y2, 100)

plt.plot(xk, yf, color='blue')
plt.plot(xk1, yf1, color='green')
plt.plot(xk2, yf2, color='red')
plt.xlabel('Interações:')
plt.ylabel('||F||')
plt.show()