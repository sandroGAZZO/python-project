
"""
TP6
GAZZO Sandro
"""

#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 10:20:12 2018

@author: Ana
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as LA
import random

"""
 TP4 : Fonctions qui implémentent les méthodes de résolution d'EDO

============================================

 *********************************
 Méthodes explicites
 *********************************

 Euler explicite
 ====================
 - f est une fonction qui prend un paramètre vecteur colonne
    et renvoie un vecteur colonne de la même taille
 - On subdivise l'intervalle [0,T] à N sous-intervalles
 - La condition initiale en 0 est le vecteur colonne U0
 """
def EulerExplicite(f,T,N,U0):
    U=np.zeros((np.size(U0),N+1))
    U[:,0]=U0
    h=T/N
    for i in range(1,N+1):
        U[:,i]=U[:,i-1]+h*f(U[:,i-1])
    return U

"""
 Euler amélioré
 ====================
Les paramètres sont les mêmes que pour Euler explicite
"""
def EulerAmeliore(f,T,N,U0):
    U=np.zeros((np.size(U0),N+1))
    U[:,0]=U0
    h=T/N
    for i in range(1,N+1):
        U[:,i]=U[:,i-1]+h*(f(U[:,i-1])+f(U[:,i-1]+h*f(U[:,i-1])))/2
    return U



"""
 Heun
 ====================
 Les paramètres sont les mêmes que pour Euler explicite
    """
def Heun(f,T,N,U0):
    U=np.zeros((np.size(U0),N+1))
    U[:,0]=U0
    h=T/N
    for i in range(1,N+1):
        U[:,i]=U[:,i-1]+h*((1/4)*f(U[:,i-1])+(3/4)*f(U[:,i-1]+(2/3)*h*f(U[:,i-1]+(h/3)*f(U[:,i-1]))))
    return U
  
"""
 Runge-Kutta
====================
 - f est une fonction qui prend un paramètre vecteur colonne
   et renvoie un vecteur colonne de la même taille
 - On subdivise l'intervalle [0,T] à N sous-intervalles
 - La condition initiale en 0 est le vecteur colonne U0
"""
def RungeKutta(f,T,N,U0):
    h = T/N
    U=np.zeros([U0.size,N+1])
    U[:,0]= U0
    for n in range(N):
        K1 = f(U[:,n]);
        K2 = f(U[:,n]+h*K1/2)
        K3 = f(U[:,n]+h*K2/2)
        K4 = f(U[:,n]+h*K3);
        U[:,n+1] = U[:,n]+h*(K1+2*K2+2*K3+K4)/6
    return U
    

"""
 *********************************
 Méthodes implicites
 *********************************

 La méthode de Newton qui est utilisée pour résoudre
 les équations nécessaires aux méthodes implicites
 """
def Newton(ff,Dff,x0):
    eps = 10**(-10)
    nmax=40
    val = x0
    i=0
    while (i<= nmax and LA.norm(ff(val))>eps):
        val = val - ff(val)/Dff(val)
        i=i+1
    return val

def NewtonV(ff,Dff,x0):
    eps = 10**(-10)
    nmax=40
    val = x0
    i=0
    while (i<= nmax and LA.norm(ff(val))>eps):
        val = val - LA.solve(Dff(val),ff(val))
        i=i+1
    return val

"""
 Euler implicite
 ====================
- f est une fonction qui prend un paramètre vecteur colonne
   et renvoie un vecteur colonne de la même taille
- Df est la fonction dérivée de f (ou la jacobienne dans le cas de R^n),
   nécessaire pour la méthode de Newton
- On subdivise l'intervalle [0,T] à N sous-intervalles
- La condition initiale en 0 est le vecteur colonne U0
"""
def EulerImplicite(f,Df,T,N,U0):
      h = T/N
      U=np.zeros([U0.size,N+1])
      U[:,0] = U0
      def Dff(x):
          return h*Df(x)-np.eye(U0.size)
      for n in range(N):
          def ff(x):
              uu=U[:,n]+h*f(x)-x
              return uu
          U[:,n+1]= NewtonV(ff,Dff,U[:,n]) 
          # U(n+1) est la solution par Newton de X = U(n)+h*f(X);
      return U
"""
Crank-Nicolson
====================
les paramètres sont les mêmes que pour Euler implicite
"""
def CrankNicolson(f,Df,T,N,U0):
    h = T/N
    U=np.zeros([U0.size,N+1])
    U[:,0] = U0
    def Dff(x):
        return h*Df(x)/2-np.eye(U0.size)
    for n in range(N):
        def ff(x):
            return U[:,n]+h*(f(U[:,n])+f(x))/2-x
        U[:,n+1]= NewtonV(ff,Dff,U[:,n])
    return U
#%%
"""
Exercice 1
"""
"""
Questions b, c et d
"""
def solex(t):
    return np.exp(-10*t)*2

def f(x):
    return -10*x

t=np.linspace(0,1,100)
plt.figure(1)
plt.plot(t,solex(t),label="solution exacte")
for k in [4,5,6,7,8,9,10]:
    i=np.linspace(0,1,2**k+1)
    plt.plot(i,EulerExplicite(f,1,2**k,2)[0,:])
plt.title("Solution exacte et approchée par Euler Explicite")
plt.legend()

plt.figure(2)
err=[]
for k in [4,5,6,7,8,9,10]:
    i=np.linspace(0,1,2**k+1)
    err=err+[LA.norm(solex(i)-EulerExplicite(f,1.,2**k,2)[0,:],np.inf)]
plt.yscale("log")
plt.xscale("log")
plt.plot([2**(-k) for k in [4,5,6,7,8,9,10]],np.array(err),label="courbe d'erreur")
plt.plot(t,t,label="droite de pente 1")
plt.title("Erreur de la méthode Euler explicite")
plt.legend()

"""
L'erreur a quasiment une pente 1. C'est un résultat moyen.
"""
"""
Question e
"""
def solex1(t):
    return np.exp(3*t)*(1/3)

def solex2(t):
    return np.exp(-3*t)*(1/3)

def f1(x):
    return 3*x

def f2(x):
    return -3*x

epsi=random.random()*10**(-7)

t20=np.linspace(0,20,500)
plt.figure(20)
plt.plot(t20,solex1(t20),label="solution exacte")
plt.plot(t20,EulerExplicite(f1,1.,499,np.array([1/3+epsi]))[0,:],label="solution approchée avec perturbation")
plt.title("Graphe de solex1")
plt.legend()
plt.figure(21)
plt.plot(t20,solex2(t20),label="solution exacte")
plt.plot(t20,EulerExplicite(f2,1.,499,np.array([1/3+epsi]))[0,:],label="solution approchée avec perturbation")
plt.title("Graphe de solex2")
plt.legend()
"""
La perturbation nous fait obtenir une erreur énorme pour solex1, un peu plus admissible pour solex2.
"""


"""
Question f
"""

def f3(x):
    return -x

plt.figure(64)
color=["red","purple","green"]
N_list=["solution pour N=6","solution pour N=14","solution pour N=60"]
cmp=0
for N in [6,14,60]:
    i=np.linspace(0,30,N+1)
    plt.plot(i,EulerExplicite(f3,30,N,1)[0,:],color[cmp],label=N_list[cmp])
    cmp=cmp+1
plt.title("A-stabilité")
plt.legend()

plt.figure(5)
cmp=0
plt.axis([0,30,-10,10])
for N in [6,14,60]:
    i=np.linspace(0,30,N+1)
    plt.plot(i,EulerExplicite(f3,30,N,1)[0,:],color[cmp],label=N_list[cmp])
    cmp=cmp+1
plt.title("Zoom sur la A-stabilité")
plt.legend()
    
#%%
"""
Exercice 2
"""
def solex5(t):
    return np.exp(-t)*2
def f5(x):
    return -x
def df5(x):
    return -1.

erEE=[]
erEA=[]
erHE=[]
erEI=[]
erCN=[]
erRK=[]
hh=[]
er1=[]
t=np.linspace(0,5,100)
plt.figure(12)
for k in [4,5,6,7,8,9,10]:
    i=np.linspace(0,5,2**k+1)
    YE=solex5(i)
    
    YEE=EulerExplicite(f5,5,2**k,np.array([2.]))
    erEE.append(LA.norm(YE-YEE[0,:],np.inf))
    
    YEA=EulerAmeliore(f5,5,2**k,np.array([2.]))
    erEA.append(LA.norm(YE-YEA[0,:],np.inf))
    
    YHE=Heun(f5,5,2**k,np.array([2.]))
    erHE.append(LA.norm(YE-YHE[0,:],np.inf))
    
    YEI=EulerImplicite(f5,df5,5,2**k,np.array([2.]))
    erEI.append(LA.norm(YE-YEI[0,:],np.inf))
    
    YCN=CrankNicolson(f5,df5,5,2**k,np.array([2.]))
    erCN.append(LA.norm(YE-YCN[0,:],np.inf))
    
    YRK=RungeKutta(f5,5,2**k,np.array([2.]))
    erRK.append(LA.norm(YE-YRK[0,:],np.inf))
    
    hh.append(5*2**(-k))

plt.yscale("log")
plt.xscale("log")
plt.plot(np.array(hh),np.array(erEE),label="erreur Euler Explicite")
plt.plot(np.array(hh),np.array(erEA),label="erreur Euler Amélioré")
plt.plot(np.array(hh),np.array(erHE),label="erreur Heun")
plt.plot(np.array(hh),np.array(erEI),label="erreur Euler Implicite")
plt.plot(np.array(hh),np.array(erCN),label="erreur Crank-Nicolson")
plt.plot(np.array(hh),np.array(erRK),label="erreur Runge-Kutta")
plt.plot(np.array(hh),np.array(hh),label="droite de pente 1")
plt.plot(np.array(hh),np.array(hh)**2,label="droite de pente 2")
plt.plot(np.array(hh),np.array(hh)**4,label="droite de pente 4")
plt.legend()
#%%
"""
Exercice 3
"""
lam=-100
mu=-1
t20=np.linspace(0,20,2001)
JAC=np.array([[0,1],[-mu*lam,mu+lam]])
def solexacte(t):
    return LA.expm(JAC*t).dot(np.array([[1],[1]]))

def fexacte(X):
    return np.array([X[1],-mu*lam*X[0]+(mu+lam)*X[1]])
def Dsolexacte(t):
    return JAC*t
EE1=[]
EI1=[]
for N,h in [[966,0.0207],[1030,0.0194],[2000,0.01]]:
    T=h*N
    i=np.linspace(0,20,N+1)
    EE=EulerExplicite(fexacte,T,N,np.array([1,1]))
    EI=EulerImplicite(fexacte,Dsolexacte,T,N,np.array([1,1]))
plt.figure(70)
plt.plot(EI,EE)
plt.figure(71)
plt.plot(t20,EE[0,:])
plt.figure(72)
plt.plot(t20,EE[1,:])
plt.figure(73)
plt.plot(t20,EI[0,:])
plt.figure(74)
plt.plot(t20,EI[1,:])
plt.figure(79)
plt.plot(EI[0,:],EE[0,:])
plt.figure(78)
plt.plot(EI[1,:],EE[1,:])
plt.figure(77)
plt.plot(np.array([t20,t20]),EE)
plt.figure(80)
plt.plot(np.array([t20,t20]),EI)

#%%
"""
Exercice 4
"""
"""
Question a)
"""
a=3.
b=1.
c=2.
d=1.
x0=1.
y0=2.
T=10.
N=200
plt.figure(6)
def F(X):
    return np.array([a*X[0]-b*X[0]*X[1],-c*X[1]+d*X[0]*X[1]])
u=np.linspace(0,T,N+1)
v=np.array([x0,y0])
EE=EulerExplicite(F,T,N,v)
plt.plot(EE[0,:],EE[1,:])
plt.scatter(x0,y0)
plt.title("Euler Explicite")

def JacF(X):
    return np.array([[a-b*X[1],d*X[1]],[-b*X[0],-c+d*X[0]]])
plt.figure(7)
EI=EulerImplicite(F,JacF,T,N,v)
plt.plot(EI[0,:],EI[1,:])
plt.title("Euler Implicite")
plt.scatter(x0,y0)
plt.figure(11)
EA=EulerAmeliore(F,T,N,v)
plt.plot(EA[0,:],EA[1,:])
plt.title("Euler Améliorée")
plt.scatter(x0,y0)
plt.figure(8)
CN=CrankNicolson(F,JacF,T,N,v)
plt.plot(CN[0,:],CN[1,:])
plt.title("Crank-Nicolson")
plt.scatter(x0,y0)
plt.figure(9)
RK=RungeKutta(F,T,N,v)
plt.plot(RK[0,:],RK[1,:])
plt.scatter(x0,y0)
plt.title("Runge-Kutta")
plt.figure(10)
He=Heun(F,T,N,v)
plt.plot(He[0,:],He[1,:])
plt.title("Heun")
plt.scatter(x0,y0)

"""
Question c
"""
T2=400
N2=2000
plt.figure(15)
EA=EulerAmeliore(F,T2,N2,v)
plt.plot(EA[0,:],EA[1,:])
plt.title("Euler Améliorée (c)")
plt.scatter(x0,y0)
plt.figure(16)
He=Heun(F,T2,N2,v)
plt.plot(He[0,:],He[1,:])
plt.title("Heun (c)")
plt.scatter(x0,y0)
plt.figure(17)
CN=CrankNicolson(F,JacF,T2,N2,v)
plt.plot(CN[0,:],CN[1,:])
plt.title("Crank-Nicolson (c)")
plt.scatter(x0,y0)
plt.figure(18)
RK=RungeKutta(F,T2,N2,v)
plt.plot(RK[0,:],RK[1,:])
plt.scatter(x0,y0)
plt.title("Runge-Kutta (c)")
"""
Question d
"""
N3=20000
plt.figure(19)
EE=EulerExplicite(F,T,N3,v)
plt.plot(EE[0,:],EE[1,:])
plt.scatter(x0,y0)
plt.title("Euler Explicite (d)")
