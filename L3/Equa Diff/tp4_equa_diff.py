""""
TP4
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
#%%
a=0.5
b=0.2
K1=100
K2=50
t=np.linspace(0,25,500)
x1=np.linspace(0,250,20)
y1=np.linspace(0,250,20)
def F(X,t):
    return [X[0]*(K1-X[0]-a*X[1])/K1,X[1]*(K2-X[1]-b*X[0])/K2]
plt.figure(1)
XX,YY=np.meshgrid(x1,y1)
UU,VV=F(np.meshgrid(x1,y1),0)
M=np.hypot(UU,VV)
M[M==0]=1
UU/=M
VV/=M
plt.quiver(XX,YY,UU,VV,angles="xy")
plt.plot([0,K1],[K1/a,0],linewidth=1.5,color='blue')
plt.plot([0,K2/b],[K2,0],linewidth=1.5,color='red')
plt.plot([0,K2/b],[0,0],linewidth=1.5,color='red')
plt.plot([0,0],[K1/a,0],linewidth=1.5,color='blue')
plt.plot((K1-a*K2)/(1-a*b),(K2-b*K1)/(1-a*b),marker='o',color='orange')
plt.plot(0,0,marker='o',color='orange')
plt.plot(0,K2,marker='o',color='orange')
plt.plot(K1,0,marker='o',color='orange')
c_i=[[250,5],[250,100],[250,50],[250,200],[5,250],[100,250],[200,250],[50,250],[5,5]]
for c,d in c_i:
    v=odeint(F,[c,d],t)
    plt.plot(v[:,0],v[:,1])
#%%
a=1
b=1
K1=100
K2=50
t=np.linspace(0,25,500)
x1=np.linspace(0,250,20)
y1=np.linspace(0,250,20)
def F(X,t):
    return [X[0]*(K1-X[0]-a*X[1])/K1,X[1]*(K2-X[1]-b*X[0])/K2]
plt.figure(2)
XX,YY=np.meshgrid(x1,y1)
UU,VV=F(np.meshgrid(x1,y1),0)
M=np.hypot(UU,VV)
M[M==0]=1
UU/=M
VV/=M
plt.quiver(XX,YY,UU,VV,angles="xy")
c_i=[[250,5],[250,100],[250,50],[250,200],[5,250],[100,250],[200,250],[50,250],[5,5]]
plt.plot([0,K1],[K1/a,0],linewidth=1.5,color='blue')
plt.plot([0,K2/b],[K2,0],linewidth=1.5,color='red')
plt.plot([0,250],[0,0],linewidth=1.5,color='red')
plt.plot([0,0],[250,0],linewidth=1.5,color='blue')
plt.plot(0,0,marker='o',color='orange')
plt.plot(0,K2,marker='o',color='orange')
plt.plot(K1,0,marker='o',color='orange')
for c,d in c_i:
    v=odeint(F,[c,d],t)
    plt.plot(v[:,0],v[:,1])
