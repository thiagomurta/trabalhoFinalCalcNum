import matplotlib.pyplot as plt
import numpy as np

def g(x):
    return ((np.pi**2) * np.sin(np.pi * x)) - 2 * (np.e**(np.pi*x))

def Jacobiana(x):
    j = np.zeros([7, 7])
    
    for i in range(0, 6):
        j[i, i+1] = -1
        j[i+1, i] = -1
    
    j[0,0] = 3*(x[1]**2)
    j[1,1] = 3*(x[1]**2)
    j[2,2] = 2*(x[1]**2)
    j[3,3] = 3*(x[1]**2)
    j[4,4] = 3*(x[1]**2)
    j[5,5] = 3*(x[1]**2)
    j[6,6] = 3*(x[1]**2)
    return j

def residuo(x):
    F = np.zeros([7,1])
    F[0] = (x[0]**2) + x[1]*x[0] - 10
    F[1] = x[1] + 3*x[0]*(x[1]**2) - 57
    return F

def solveSystem(x, tol, kmax):
    k = 0
    F = residuo(x)
    while(np.linalg.norm(F) > tol and k < kmax):
        j = Jacobiana(x)
        dx = np.linalg.solve(j, -F)
        x = x + dx
        F = residuo(x)
        k = k+1
    return k, F

x = np.array([[-1], [3.5]])

x = np.array([ [1.5], [3.5]])
tol = 10e-4

n = 10
a = 1.
b = 6.
k = 0
tolerance = 10e-7
xk = np.linspace(a, b, n)
h = (b-a)/n