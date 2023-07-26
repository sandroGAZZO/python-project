"""
TP Noté
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
#%%
"""
Partie I
"""
a=0.5
b=0.02
c=0.2
d=0.01
e=0.01
N=20
t=np.linspace(-5,25,500)
x=np.linspace(-5,5,20)
y=np.linspace(0,40,20)
def F(X,t):
    return a*X
plt.figure(1)
plt.axis([-5,5,0,40])
XX,YY=np.meshgrid(x,y)
U=np.ones([N,N])
V=F(YY,XX)
M=np.hypot(U,V)
M[M==0]=1
U/=M
V/=M
plt.quiver(x,y,U,V,angles="xy")
c_i=[0.3,1,5,15]
for u in c_i:
    v=odeint(F,u,t)
    plt.plot(t,v)
plt.xlabel("Portrait de phase de (1)")

def G(X,t):
    return c*X-d*(X**2)
plt.figure(2)
plt.axis([-5,5,0,40])
XX,YY=np.meshgrid(x,y)
U=np.ones([N,N])
V=G(YY,XX)
M=np.hypot(U,V)
M[M==0]=1
U/=M
V/=M
plt.quiver(x,y,U,V,angles="xy")
c_i2=[3,5,15,25,39,10]
for u in c_i2:
    v=odeint(G,u,t)
    plt.plot(t,v)
plt.xlabel("Portrait de phase de (2)")

"""
La population (1) va croître exponentiellement alors que la population (2) va se stabiliser.
C'est un résultat attendu de part les équations que l'on nous a donné.
"""

#%%
"""
Partie II
"""
a=0.5
b=0.02
c=0.2
d=0.01
e=0.01
t=np.linspace(0,25,500)
x1=np.linspace(0,40,20)
y1=np.linspace(0,40,20)
def F1(X,t):
    return [X[0]*a-b*X[0]*X[1],c*X[1]-d*(X[1]**2)-e*X[0]*X[1]]
plt.figure(3)
plt.axis([0,40,0,40])
JJ,SS=np.meshgrid(x1,y1)
UU,VV=F1(np.meshgrid(x1,y1),0)
M2=np.hypot(UU,VV)
M2[M2==0]=1
UU/=M2
VV/=M2
plt.quiver(JJ,SS,UU,VV,angles="xy")
plt.plot([0.1,0.1],[0,c/d],linewidth=1.5,color='blue')
plt.plot([0,40],[0,0],linewidth=1.5,color='red')
plt.plot(0.1,0.1,marker='o',color='orange') #le point d'équilibre est en (0,0) mais pour le voir mieux j'ai mis (0.1,0.1)
plt.plot(0.1,c/d,marker='o',color='orange') #le point d'équilibre est en (0,0) mais pour le voir mieux j'ai mis (0.1,c/d)
c_i3=[[5,40],[10,40],[25,40],[17,40],[35,40],[0.1,5],[0.1,10],[0.3,35],[0.1,1]]
for o,p in c_i3:
    v=odeint(F1,[o,p],t)
    plt.plot(v[:,0],v[:,1])

plt.xlabel("Portrait de phase de (3)")






