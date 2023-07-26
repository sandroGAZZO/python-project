#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
#%%
"""
Exercice 0
"""
t=np.linspace(-5,5,100)
x=t**2
y=t*np.sin(2*t)
dessin=Axes3D(plt.figure())
dessin.plot(t,x,y) 
dessin.legend()
dessin.set_xlabel('t')
#%%
"""
Exercice 1
"""
plt.figure(1)
t=np.linspace(-4*np.pi,4*np.pi,100)
x=np.cos(t)
y=-np.sin(t)
for i in [1,2,3,4]:
    x1=i*x
    y1=i*y
    plt.plot(t,x1)
    plt.plot(t,y1,'--')

plt.figure(2)
dessin=Axes3D(plt.figure())
for i in [1,2,3,4]:
    x1=i*x
    y1=i*y
    dessin.plot(t,x1,y1)
dessin.set_xlabel('t')

plt.figure()
def F(X,t):
    return [X[1],-X[0]]
N=20
u=np.linspace(-4,4,N)
v=np.linspace(-4,4,N)
XX,YY=np.meshgrid(u,v)
UU,VV=F(np.meshgrid(u,v),0)
M=np.hypot(UU,VV)
M[M==0]=1
UU/=M
VV/=M
plt.quiver(XX,YY,UU,VV,angles="xy")

for i in [1,2,3,4,]:
    x1=i*x
    y1=i*y
    plt.plot(x1,y1)
    
plt.show()

#%%
"""
Exercice 2
"""
axes=plt.gca()
def TracePhase(choix_constantes):
    t=np.linspace(-3,3,100)
    for i in choix_constantes:
        x=np.exp((t**2)/2)*(i[0]*np.exp(t)+i[1]*np.exp(-t))
        y=np.exp((t**2)/2)*(-i[0]*np.exp(t)-2*i[1]*np.exp(-t))
        axes.set_xlim(0,15)
        axes.set_ylim(-30,0)
        plt.plot(x,y)
choix_constantes=[[1,1],[1,5],[3,2],[2,2]]
TracePhase(choix_constantes)

"""
Les courbes s'intersectent. Il faut le représenter en 3D.
"""
#%%
"""
Interlude
"""
def F(X,t):
    return [X[0],-2*X[1]]
t=np.linspace(-100,100,1000)
condition_initiale=[[1,4],[-0.5,3],[1,-3],[-1,-5]]
plt.axis([-10,10,-3,3])
for a,b in condition_initiale:
    v=odeint(F,[a,b],t)
    CI="x(0)={0},y(0)={1}".format(a,b)
    plt.plot(v[:,0],v[:,1],label=CI)
plt.legend()

#%%
"""
Exercice 3
"""
def F(X,t):
    return [X[1],-2*X[0]-3*X[1]]
plt.figure(1)
N=20
x=np.linspace(-5,5,N)
y=np.linspace(-5,5,N)
XX,YY=np.meshgrid(x,y)
UU,VV=F(np.meshgrid(x,y),0)
M=np.hypot(UU,VV)
M[M==0]=1
UU/=M
VV/=M
plt.quiver(XX,YY,UU,VV,angles="xy")

t=np.linspace(-10,10,1000)
c_i=[[1,2],[-2,-2],[0,-4],[-4,0.5],[3,4]]
plt.axis([-5,5,-5,5])
for a,b in c_i:
    v=odeint(F,[a,b],t)
    CI="x(-10)={0},y(-10)={1}".format(a,b)
    plt.plot(v[:,0],v[:,1],label=CI)
plt.legend()

plt.plot([-5,5],[5,-5])
plt.plot([-2.5,2.5],[5,-5])



#%%
"""
Exercice 4
"""
def f(x):
    return np.exp(-1/x)*(x>0)
plt.figure(1)
t=np.linspace(-5,5,100)
plt.plot(t,f(t))

def F(X,t):
    return [X[1]-X[0]*f(X[1]**2+X[0]**2),-X[0]-X[1]*f(X[0]**2+X[1]**2)]

plt.figure(2)
t=np.linspace(-10,10,1000)
axes=plt.gca()
N=20
x=np.linspace(-3,3,N)
y=np.linspace(-3,3,N)
XX,YY=np.meshgrid(x,y)
UU,VV=F(np.meshgrid(x,y),0)
M=np.hypot(UU,VV)
M[M==0]=1
UU/=M
VV/=M
plt.quiver(XX,YY,UU,VV,angles="xy")

c_i=[]
for i in range(-10,11):
    c_i=c_i+[[i,i]]
plt.axis([-5,5,-5,5])
for a,b in c_i:
    axes.set_xlim(-3,3)
    axes.set_ylim(-3,3)
    v=odeint(F,[a,b],t)
    plt.plot(v[:,0],v[:,1])
    
#%%
"""
Interlude
"""
def g(x,y):
    return (x**2+7*(y**2)*(1+np.arctan(np.sin(x))))
N=1000
x=np.linspace(-4,4,N)
y=np.linspace(-3,3,N)
xx,yy=np.meshgrid(x,y)
zz=g(xx,yy)
C=np.linspace(1,15,5)
lignes_niveaux=plt.contour(xx,yy,zz,levels=C)
plt.clabel(lignes_niveaux,C,fmt='%1.0f')

#%%
"""
Exercice 5
"""
def F(X,t):
    return [X[1],-X[0]-X[0]**3]

plt.figure(1)
N=20
x=np.linspace(-10,10,N)
y=np.linspace(-10,10,N)
XX,YY=np.meshgrid(x,y)
UU,VV=F(np.meshgrid(x,y),0)
M=np.hypot(UU,VV)
M[M==0]=1
UU/=M
VV/=M
plt.quiver(XX,YY,UU,VV,angles="xy")

def g(x,y):
    return 2*(x**2)+x**4+2*(y**2)
zz=g(XX,YY)
C=[10,80,180]
lignes_niveaux=plt.contour(XX,YY,zz,levels=C)
plt.clabel(lignes_niveaux,C,fmt='%1.0f')

t=np.linspace(-10,10,1000)
c_i=[[-2,2],[-1,0.5],[3,4]]
for a,b in c_i:
    v=odeint(F,[a,b],t)
    CI="x(-10)={0},y(-10)={1}".format(a,b)
    plt.plot(v[:,0],v[:,1],label=CI)
plt.legend()

