
"""
CIMETTA Marvin
GAZZO Sandro
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as LA
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import CubicSpline
#%%
"""
Exercice1
"""
def B(x,k,i,t):
    assert len(t)>(k-1)
    c=np.zeros(len(x))
    if k==0:
        for l in range (len(x)):
            if t[i]<=x[l]<t[i+1]:
                c[l]=1
    else:
        c=((x-t[i])/(t[i+k]-t[i]))*B(x,k-1,i,t)+((t[i+k+1]-x)/(t[i+k+1]-t[i+1]))*B(x,k-1,i+1,t)
    return c

def BsplineF(x,t,c,k):
    assert len(t)>(k-1)
    som=0
    for i in range (len(x)):
        som+=c[i]*B(x,k,i,t)
    return som

def DerBspline(x,k,i,t):
    assert len(t)>(k-1)
    assert 1<=k
    return k*((B(x,k-1,i,t)/(t[i+k]-t[i]))-(B(x,k-1,i+1,t)/(t[i+k+1]-t[i+1])))

#%%
"""
Exercice 2
"""
def MCoeff3(x,t,n):
    m=np.zeros([len(x),len(x)])
    for d in range (len(x)):
        for l in range (len(x)):
            m[d,l]=B([x[l]],3,d,t)
    return m
    
def f0(x):
    return np.sin(x)

#%% 
n=10
k=3
t=np.arange(n)
x=t[k:(n-k)]
z=np.linspace(3,6,100)
print(B(x,k,1,t))
print(MCoeff3(x,t,n))
f=f0(x)
print(BsplineF(x,t,f,k))
print(DerBspline(x,k,1,t))
print(LA.solve(MCoeff3(x,t,n),DerBspline(x,k,len(x),t)))
print(LA.solve(MCoeff3(x,t,n),BsplineF(x,t,f,k)))
plt.plot(x,BsplineF(x,t,f,k))
plt.plot(z,f0(z))

"""
Qu'entend-elle par "fonction sinus"? Un vecteur x choisi auquel on lui
applique la fonction sinus et on récupère les valeurs pour les mettre dans
un nouveau vecteur?
Il faut répondre à cette question. Une fois fait, le reste de l'exo2
correspond à ce que l'on a déjà fait à part la partie avec LA.solve qui
pourra peut-être poser problème.
"""

#%%
"""
Exercice 3
"""
def ChebNodes(N):
    x=[]
    for i in range (0,N):
        x+=[-np.cos(((2*i+1)*np.pi)/(2*N))]
    return x

def f1(x):
    c=np.zeros(len(x))
    for i in range (len(x)):
        c[i]=np.sqrt(1-((x[i])**2))
    return c

#%%
"""
Question b
"""
x=ChebNodes(50)
s=InterpolatedUnivariateSpline(x,f1(x))
plt.plot(x,f1(x))
plt.plot(x,s(x))
#%%
x=ChebNodes(100)
s=InterpolatedUnivariateSpline(x,f1(x))
plt.plot(x,f1(x))
plt.plot(x,s(x))
#%%
x=ChebNodes(10)
s=InterpolatedUnivariateSpline(x,f1(x))
plt.plot(x,f1(x))
plt.plot(x,s(x))
#%%
x=ChebNodes(10)
s=UnivariateSpline(x,f1(x))
plt.plot(x,f1(x))
plt.plot(x,s(x))
#%%
x=ChebNodes(10)
s=UnivariateSpline(x,f1(x))
s.set_smoothing_factor(0.5)
plt.plot(x,f1(x))
plt.plot(x,s(x))
#%%
x=ChebNodes(100)
s=UnivariateSpline(x,f1(x))
plt.plot(x,f1(x))
plt.plot(x,s(x))
#%%
x=ChebNodes(100)
s=UnivariateSpline(x,f1(x))
s.set_smoothing_factor(0.5)
plt.plot(x,f1(x))
plt.plot(x,s(x))
#%%
x=ChebNodes(100)
s=UnivariateSpline(x,f1(x))
s.set_smoothing_factor(3)
plt.plot(x,f1(x))
plt.plot(x,s(x))
"""
On constate que rien ne change pour cette fonction si l'on change les paramètres de lissage.
En revanche, on constate que si l'on prend un nombre trop petit de points de Tchebyshev,
les splines seront moins lisses
"""
#%%
"""
Question c
"""
x=np.linspace(-3,3,50)
z=np.exp(-x**2)+0.1*np.random.randn(50)
s=InterpolatedUnivariateSpline(x,z)
plt.plot(x,z)
plt.plot(x,s(x))
#%%
w=np.linspace(-3,3,100)
x=np.linspace(-3,3,50)
z=np.exp(-x**2)+0.1*np.random.randn(50)
s=InterpolatedUnivariateSpline(x,z)
plt.plot(x,z)
plt.plot(w,s(w))
#%%
w=np.linspace(-3,3,10)
x=np.linspace(-3,3,50)
z=np.exp(-x**2)+0.1*np.random.randn(50)
s=InterpolatedUnivariateSpline(x,z)
plt.plot(x,z)
plt.plot(w,s(w))
#%%
w=np.linspace(-3,3,500)
x=np.linspace(-3,3,50)
z=np.exp(-x**2)+0.1*np.random.randn(50)
s=InterpolatedUnivariateSpline(x,z)
plt.plot(x,z)
plt.plot(w,s(w))
#%%
x=np.linspace(-3,3,50)
z=np.exp(-x**2)+0.1*np.random.randn(50)
s=UnivariateSpline(x,z)
plt.plot(x,z)
plt.plot(x,s(x))
#%%
w=np.linspace(-3,3,100)
x=np.linspace(-3,3,50)
z=np.exp(-x**2)+0.1*np.random.randn(50)
s=UnivariateSpline(x,z)
plt.plot(x,z)
plt.plot(w,s(w))
#%%
w=np.linspace(-3,3,100)
x=np.linspace(-3,3,50)
z=np.exp(-x**2)+0.1*np.random.randn(50)
s=InterpolatedUnivariateSpline(x,z)
s.set_smoothing_factor(0.5)
plt.plot(x,z)
plt.plot(w,s(w))
#%%
w=np.linspace(-3,3,100)
x=np.linspace(-3,3,50)
z=np.exp(-x**2)+0.1*np.random.randn(50)
s=InterpolatedUnivariateSpline(x,z)
s.set_smoothing_factor(0.3)
plt.plot(x,z)
plt.plot(w,s(w))
"""
On constate cette fois ci un changement de la courbe si l'on prend ou moins de point au départ.
De même que précédemment, plus on aura de noeuds pour la spline, plus elle sera lisse.
En revanche, ici, le paramètre de degré de lissage fait changer la spline. Plus le paramètre
est petit, plus la spline sera proche de la fonction de départ.
"""
#%%
"""
Question d
"""
x=np.linspace(-2*np.pi,2*np.pi,6)
s=CubicSpline(x,f0(x))
t=np.linspace(-2*np.pi,2*np.pi,100)
plt.plot(t,f0(t))
plt.plot(x,s(x))
#%%
x=np.linspace(-2*np.pi,2*np.pi,15)
s=CubicSpline(x,f0(x))
t=np.linspace(-2*np.pi,2*np.pi,100)
plt.plot(t,f0(t))
plt.plot(x,s(x))
#%%
x=np.linspace(-2*np.pi,2*np.pi,6)
s=CubicSpline(x,f0(x))
t=np.linspace(-2*np.pi,2*np.pi,100)
z=np.arange(-2*np.pi,2*np.pi,0.1)
plt.plot(t,f0(t),label='sin(x)')
plt.plot(z,s(z),label='s')
plt.plot(z,s(z,1),label="s'")
plt.plot(z,s(z,2),label="s''")
plt.plot(z,s(z,3),label="s'''")
plt.legend(ncol=3)
plt.show()