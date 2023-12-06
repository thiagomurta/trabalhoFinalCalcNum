import sympy as sy
from sympy import init_printing
init_printing(use_latex='png', scale=1.05, order='grlex', forecolor='Black', backcolor='White', fontsize=10)
from sympy import diff, symbols, sin, cos, exp, Pow
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import scipy.io
from scipy.linalg import lu
import scipy.sparse.linalg


def Jacobiana(x):
    j = np.zeros([2,2])
    j[0,0] = 2*x[0]+x[1]
    j[0,1] = x[0]
    j[1,0] = 3*(x[1]**2)
    j[1,1] = 1 + 6*x[0]*x[1]
    return j

def residuo(x):
    F = np.zeros([2,1])
    F[0] = (x[0]**2) + x[1]*x[0] - 10
    F[1] = x[1] + 3*x[0]*(x[1]**2) - 57
    return F

def solveSystem(x, tol, kmax):
    k = 0
    F = residuo(x)
    while(np.linalg.norm(F) > tol and k < kmax):
        print("x0: ", x)
        j = Jacobiana(x)
        print("j: ",j)
        dx = np.linalg.solve(j, -F)
        print("Dx: ", dx)
        x = x + np.array(np.transpose(dx))
        print("x1: ", x)
        F = residuo(x)
        k = k+1
    return k, F

x = [1.5, 3.5]
tol = 10e-4

tamMax = 100
k, F = solveSystem(x, tol, tamMax)
#plt.xlabel(t)
#plt.ylabel(np.linalg.norm(F))
#plt.show()

#FEniCSx

#### PVC
'''
n = 10
a = 1.
b = 6.
k = 0
tolerance = 10e-7
xk = np.linspace(a, b, n)
h = (b-a)/n
'''