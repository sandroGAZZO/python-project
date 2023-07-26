"""
GAZZO Sandro
CIMETTA Marvin
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy import optimize
#%%
"""
Exercice1
"""
A=np.random.rand(30,50)*20-10
s=linalg.svdvals(A)
At=np.transpose(A)
s_trans=sorted(np.real(linalg.eigvals(At.dot(A))),reverse=True)
s2=sorted(np.real(linalg.eigvals(A.dot(At))),reverse=True)
s1=[s_trans[i] for i in range (0,len(s))]
t=[i for i in range (0,len(s))]
AA=np.zeros((80,80))
AA[:30,30:]=A
AA[30:,:30]=At
s3=sorted(np.real(linalg.eigvals(AA)),reverse=True)
s3_30=s3[:30]
plt.figure(0)
plt.plot(t,s2)
plt.plot(t,s*s)
plt.plot(t,s1)
plt.plot(t,s3_30)
"""
Toutes les courbes sont confondues, sauf pour la matrice AA.
"""

U,s_d,Vh=linalg.svd(A)
U_dix=U[:,:10]
s_d_dix=[s_d[i] for i in range(0,10)]
S=np.diag(s_d_dix)
Vh_dix=Vh[:10,:]
B=(U_dix.dot(S)).dot(Vh_dix)
opt=linalg.norm(B,'fro')
print(opt)
"""
L'erreur est grande.
"""

#%%
"""
Exercice 2
"""
fichier=open('george.dat',"r")
MP=np.genfromtxt('george.dat')
plt.figure(1)
plt.imshow(MP,cmap='gray',vmin=0,vmax=255)
plt.figure(2)
plt.imshow(MP,cmap='gray',vmin=0,vmax=5)
plt.figure(3)
plt.imshow(MP,cmap='gray',vmin=0,vmax=25)
plt.figure(4)
plt.imshow(MP,cmap='gray',vmin=0,vmax=75)
"""
Plus les rangs des approximations sont grands, plus l'image est précise
"""
#%%
"""
Exercice 3
"""
fichier=open('temperatures.dat',"r")
temp=np.genfromtxt('temperatures.dat')
def vandermonde (m,n):
    A=np.zeros((m,n))
    for i in range (1,m+1):
        for j in range (n):
            A[i-1,j]=i**j
    return A
A=vandermonde(30,3)
U,s,Vt=linalg.svd(A)
rg=np.linalg.matrix_rank(A)
U_r=U[:rg,:rg]
Utr=np.transpose(U_r)
V=np.transpose(Vt)
SUM=np.zeros((3,30))
for i in range (rg):
    SUM+=(Utr[:,i:i+1]*temp*V[:,i:i+1])/s[i]
apptemp=A.dot(SUM)+temp
plt.figure(5)
t=[i for i in range (0,len(temp))]
plt.plot(t,temp,label="Températures relevées à Villeneuve d'Ascq en Janvier 2015")
plt.plot(t,apptemp[:,:1],label="Approximation des températures rapprochées par une parabole"    )
plt.figure(6)
plt.plot(t,temp)
plt.plot(t,apptemp)
"""
La parabole respecte plutôt bien les températures données. Cela approxime bien.
On remarque également que l'on a un choix de paraboles à faire. Il y a une peut-être
une optimisation à travailler de ce côté là.
"""
#%%
"""
Exercice 4
"""
def x_lambda(lam):
    U,s,Vt=linalg.svd(A)
    r=np.linalg.matrix_rank(A)
    Ut=np.transpose(U)
    b_tilde=Ut.dot(b)
    res=0
    for i in range (r):
        res+=(((s[i]*b_tilde[i])/(s[i]**2+lam))**2)
    res=res-(alpha**2)
    return res

def MinCarresContraintes(A,b,alpha):
    U,s,Vt=linalg.svd(A)
    r=np.linalg.matrix_rank(A)
    Ut=np.transpose(U)
    V=np.transpose(Vt)
    b_tilde=Ut.dot(b)
    somme=0
    x=0
    for i in range (r):
        somme+=(b_tilde[i]/s[i])**2
    if somme > (alpha**2):
        for i in range(r):
            lambda_tilde=optimize.root(x_lambda,0)
            x+=((s[i]*b_tilde[i]*V[:,i:i+1])/((s[i]**2)+lambda_tilde.x))
    else:
        for i in range(r):
            x+=(b_tilde[i]*V[:,i:i+1])/s[i]
    return x

A=np.array([[1,2,3],[-1,3,6],[2,6,-3],[-2,0,5]])
b=np.array([4,-1,3,0])
alpha=2
MinCarresContraintes(A,b,alpha)