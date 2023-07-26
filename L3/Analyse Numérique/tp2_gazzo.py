"""
CIMETTA Marvin
GAZZO Sandro
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as LA

#%%
"""
Exercice 0
"""
x=0.5
n=3
a=0
b=1
print(min(max(np.ceil(n*(x-a)/(b-a)),1),n))
"""
Cela retourne la partie entière.
"""

def IndexInterval_unevaleur(a,b,n,x):
    return min(max(np.ceil(n*(x-a)/(b-a)),1),n)

x1=np.array([-1,0,0.5,1,2])
def IndexInterval(a,b,n,x):
    l=[]
    for i in x:
        l=l+[int(min(max(np.ceil(n*(i-a)/(b-a)),1),n))]
    return l

#%%
"""
Exercice 1
"""
def ConstPiecewise(a,b,v,x):
    n=len(v)
    k=IndexInterval(a,b,n,x)
    l=[]
    for i in k:
        l=l+[v[i-1]]
    return np.array(l)

def InterpConst(a,b,n,f):
    h=(b-a)/n
    x=np.linspace(a+h/2,b-h/2,n)
    v=f(x)
    return x,v

def f1(x):
    return np.sin(x)
#%%
plt.grid(True)
a=-0.5
b=1.5
n=1000
v=np.array([0,2,-1,1])
x=np.linspace(a,b,n)
plt.title("ConstPiecewise sur [-1/2,3/2] et hauteurs [0,2,-1,1].")
plt.plot(x,ConstPiecewise(a,b,v,x))
#%%
a=-5.
b=5.
t,v=InterpConst(a,b,10,f1)
y=np.linspace(a,b,1000)
z=ConstPiecewise(a,b,v,y)
plt.plot(y,f1(y))
plt.plot(y,z)
plt.plot(t,v,'o')
plt.title("sin(x) sur [-5,5] et son interpolation constante sur 10 intervalles")

#%%
plt.title("Erreur d'interpolation pour n=2**k (k entre 1 et 10)")
listN=np.array([2**k for k in range(1,11)])
listNorm=[]
for n in listN:
    v=np.array([f1(-5+k*(10)/n) for k in range(n+1)])
    r=ConstPiecewise(-5,5,v,x)
    z=IndexInterval(-5,5,n,x)
    listNorm.append(LA.norm(f1(x)-r,np.inf))
plt.xscale('log')
plt.yscale('log')
plt.plot(listN,np.array(listNorm),label="Erreur")
#%%
"""
Exercice 2
"""
def segment(x0,y0,x1,y1):
    lx, ly = [x0,x1], [y0,y1]
    plt.plot(lx,ly,'b')
    
def graphAffinePiecewise(a,b,v,x):
    assert len(v)>1
    n=len(v)-1
    k=IndexInterval(a,b,n,x)
    h=(b-a)/n

    for i in k :      
        segment((i-1)*h+a,v[i-1],i*h+a,v[i])

def AffinePiecewise(a,b,v,x):
    assert len(v)>1
    n=len(v)-1   
    h=(b-a)/n
    res=[]
    for z in range(len(x)):
       k=IndexInterval(a,b,n,[x[z]])
       s=k[0]
       az=(v[s]-v[s-1])/h
       bz=v[s]-az*(a+s*h)
       res.append(az*x[z]+bz)
    return res


v22=np.array([0.,2.,-1.,1.,0.])
x=np.linspace(-1/2.,3/2.,1000)
        
plt.figure(1)
y=graphAffinePiecewise(-1/2.,3/2.,v22,x)
plt.figure(2)
z=AffinePiecewise(-1/2.,3/2.,v22,x)

plt.plot(x,z) 
#%%
def InterpAffine(a,b,n,f):
    h=(b-a)/(n)
    t=[]
    for z in range(n+1):
        t.append(a+z*h)
    v=f(t)
    return t,v
       
x=np.linspace(-5.,5.,1000)
u,w=InterpAffine(-5.,5.,10,f1)
c=graphAffinePiecewise(-5.,5.,w,x)
plt.plot(x,f1(x))
plt.scatter(u,w)
#%%
plt.title("Erreur d'interpolation pour n=2**k (k entre 1 et 10)")
listN=np.array([2**k for k in range(1,11)])
listNorm=[]
for n in listN:
    v=np.array([f1(-5+k*(10)/n) for k in range(n+1)])
    r=AffinePiecewise(-5,5,v,x)
    z=IndexInterval(-5,5,n,x)
    listNorm.append(LA.norm(f1(x)-r,np.inf))
plt.xscale('log')
plt.yscale('log')
plt.plot(listN,np.array(listNorm),label="Erreur")