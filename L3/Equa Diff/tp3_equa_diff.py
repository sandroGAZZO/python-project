#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
#%%
"""
Exercice1
"""
g=10
l=10
m=1
c=0
x=np.linspace(-10,10,100)
y=np.linspace(-6,6,100)
x1=np.linspace(-10,10,20)
y1=np.linspace(-6,6,20)
xx,yy=np.meshgrid(x,y)
dessin=Axes3D(plt.figure())

def f(x,y):
    return 0.5*(y**2)+(g/l)*(1-np.cos(x))

z=f(xx,yy)
plt.figure(1)
dessin.plot_surface(xx,yy,z)

plt.figure(2)
k=[0.1,0.4,0.7,1,2,4,6,10,13,16]
plt.contour(xx,yy,z,levels=k)

def F(X,t):
    return [X[1],-(c/m)*X[1]-(g/l)*np.sin(X[0])]
XX,YY=np.meshgrid(x1,y1)
UU,VV=F(np.meshgrid(x1,y1),0)
M=np.hypot(UU,VV)
M[M==0]=1
UU/=M
VV/=M
plt.quiver(XX,YY,UU,VV,angles="xy")

plt.figure(3)
plt.quiver(XX,YY,UU,VV,angles="xy")
c_i=[[0.5,0],[0.8,0],[np.pi,-1e-6],[-np.pi,1e-6],[-10,2],[-10,4],[10,-3],[10,-5],[1.6,0],[7.5,0],[-7.5,0]]
plt.axis([-10,10,-6,6])
for a,b in c_i:
    v=odeint(F,[a,b],x)
    plt.plot(v[:,0],v[:,1])

#%%
"""
Exercice2
"""
g=10
l=10
m=1
c=1
x=np.linspace(-10,10,100)
y=np.linspace(-6,6,100)
x1=np.linspace(-10,10,20)
y1=np.linspace(-6,6,20)
plt.figure(1)
plt.axis([-10,10,-6,6])
def F(X,t):
    return [X[1],-(c/m)*X[1]-(g/l)*np.sin(X[0])]
XX,YY=np.meshgrid(x1,y1)
UU,VV=F(np.meshgrid(x1,y1),0)
M=np.hypot(UU,VV)
M[M==0]=1
UU/=M
VV/=M
plt.quiver(XX,YY,UU,VV,angles="xy")
c_i2=[[0,0],[0.5,0.5],[-0.3,-0.3],[-1,-1],[2,6],[2,2],[-7,5],[6,-4],[-4,-5]]
for a,b in c_i2:
    v=odeint(F,[a,b],x)
    plt.plot(v[:,0],v[:,1])
#%%
g=10
l=10
m=1
c=2
x=np.linspace(-10,10,100)
y=np.linspace(-6,6,100)
x1=np.linspace(-10,10,20)
y1=np.linspace(-6,6,20)
plt.figure(1)
plt.axis([-10,10,-6,6])
def F(X,t):
    return [X[1],-(c/m)*X[1]-(g/l)*np.sin(X[0])]
XX,YY=np.meshgrid(x1,y1)
UU,VV=F(np.meshgrid(x1,y1),0)
M=np.hypot(UU,VV)
M[M==0]=1
UU/=M
VV/=M
plt.quiver(XX,YY,UU,VV,angles="xy")
c_i2=[[0,0],[0.5,0.5],[-0.3,-0.3],[-1,-1],[2,6],[2,2],[-7,5],[6,-4],[-4,-5]]
for a,b in c_i2:
    v=odeint(F,[a,b],x)
    plt.plot(v[:,0],v[:,1])
#%%
g=10
l=10
m=1
c=3
x=np.linspace(-10,10,100)
y=np.linspace(-6,6,100)
x1=np.linspace(-10,10,20)
y1=np.linspace(-6,6,20)
plt.figure(1)
plt.axis([-10,10,-6,6])
def F(X,t):
    return [X[1],-(c/m)*X[1]-(g/l)*np.sin(X[0])]
XX,YY=np.meshgrid(x1,y1)
UU,VV=F(np.meshgrid(x1,y1),0)
M=np.hypot(UU,VV)
M[M==0]=1
UU/=M
VV/=M
plt.quiver(XX,YY,UU,VV,angles="xy")
c_i2=[[0,0],[0.5,0.5],[-0.3,-0.3],[-1,-1],[2,6],[2,2],[-7,5],[6,-4],[-4,-5]]
for a,b in c_i2:
    v=odeint(F,[a,b],x)
    plt.plot(v[:,0],v[:,1])
#%%
"""
Exercice 3
"""
