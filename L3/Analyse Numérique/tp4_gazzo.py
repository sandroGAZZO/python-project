
"""
CIMETTA Marvin
GAZZO Sandro
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
import math 
from scipy import linalg as LA
#%%
"""
Exercice 1
"""
def combinaison_lineaire(A,B,u,v):
    return [A[0]*u+B[0]*v,A[1]*u+B[1]*v]

def interpolation_lineaire(A,B,t):
    return combinaison_lineaire(A,B,t,1-t)


def reduction(pc,t):
    res=[]
    n=len(pc)
    for i in range(n-1):
        res+=[interpolation_lineaire(pc[i],pc[i+1],1-t)]
    return res

def point_bezier_n(pc,t):
    n=len(pc)
    while n>1:
        pc=reduction(pc,t)
        n=len(pc)
    return pc[0]

def C_Bezier(pc,N):
    n=len(pc)
    cmp=1/N
    t=cmp
    c=[pc[0]]
    while t<1:
        c+=[point_bezier_n(pc,t)]
        t+=cmp
    c+=[pc[n-1]]
    return c


#%%
u=np.linspace(0,8,100)
w=[]
v=[]
G=50
N=np.array([0,1,100])
pc=np.array([[0,0],[1,1.21],[2,-1.1],[3,0.9],[4,-1.3],[8,0.8]])
for i in range(len(C_Bezier(pc,G))):
    w+=[(((C_Bezier(pc,G)[i])[0]))]
    v+=[(((C_Bezier(pc,G)[i]))[1])]
plt.plot(w,v,'r-')
plt.plot(pc[:,0],pc[:,1],'o--')
plt.show()
#%%
"""
Exercice 2
"""
def Bt(i,n,t):
    if n==0:
        if i==0:
            return 1
        return 0
    return (1-t[:])*Bt(i,n-1,t)+t[:]*Bt(i-1,n-1,t)

n=3
t=np.linspace(0,1,1000)
plt.figure(1)
for i in range(n+1):
    plt.plot(t,Bt(i,n,t))

def Courbe_B(p,t,n):
    N=len(t)
    dim=np.size(p,0)
    d=np.size(p,1)
    P=np.zeros([p,N])
    for i in range (N):
        for j in range(d):
            P[0,i]+=P[0,i]+p[0,j]*Bt(i,n,t)
            P[1,i]+=P[1,i]+p[1,j]*Bt(i,n,t)
    return P

def Calcul_PolCon(a,n):
    V=np.zeros((n,n))
    U=V
    M=V
    for r in range (n):
        V[r,r]=(-1)*r
        U[r,r]=math.factorial(n)/(math.factorial(r)*math.factorial(n-r))
        for j in range (r):
            M[r,j]=math.factorial(r)/(math.factorial(j)*math.factorial(r-j))
    T=U*V*M*V
    return LA.solve(T,a)
 
a=[3,7,28,-8,-2,4]
n=len(a)
print(Calcul_PolCon(a,n))
e=np.array([[1,2],[4,7],[2,5],[1,6]])
print(Calcul_PolCon(e,len(e)))

def PPoly(a,t):
    N=len(t)
    d=np.size(a,1)
    P=np.zeros([a,N])
    for i in range (N):
        for j in range(d):
            P[0,i]+=P[0,i]+a[0,j]*(t**i)
            P[1,i]+=P[1,i]+a[1,j]*(t**i)
    return P

plt.figure(2)
a_final=np.array([[-1,1],[2,-2],[-3,-4],[4,2],[-1,0]])
plt.plot(Calcul_PolCon(a_final,len(a_final))[:,0],Calcul_PolCon(a_final,len(a_final))[:,1],'o--')
