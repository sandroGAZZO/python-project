##---------------------------------------##
##------------- Gazzo Sandro ------------##
##------------- Dellouve Théo -----------##
##---------------------------------------##

#importation des librairies

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

#%%
### QUESTION 3 ###

# initialisation

h=1.0 #pas dans l'espace
L=10.0 #on trace sur[-L,L]
x_m=np.linspace(-L,L,int((2*L)/h+1) ) #x discret de -L à L avec un pas de h
y_m=np.linspace(-2,2,int((2*2)/h+1) ) #x discret de -L à L avec un pas de h
x=np.linspace(-L,L,int((2*L)/h+1)*1000) #x continu


# crétaion de la grille de points :
x_grille=[] # pour définir la grille
y_grille=[] # pour définir la grille
for i in range (0,len(x_m)):
    for j in range (0,len(y_m)):
        x_grille.append(x_m[i])
        y_grille.append(y_m[j])
        
plt.scatter(x_grille,y_grille,color='red',marker='.',linewidth = 0.1)
plt.title("Grille de points")
plt.show()
        
#%%

"""
Cette fonction correspond à la fonction sinus cardinal
appliquée en un point.
"""
def Sh(h,x):
    if x==0:
        return 1
    else:
        return (np.sin(np.pi*x/h)/(np.pi*x/h))


"""
Cette fonction correpond à la fonction u1 du
sujet, plus exactement la fonction delta(x)
où delta est le symbole de Kroenecker.
Elle renvoie une liste.
"""
def u1(X,L):
    res=[]
    for x in X:
        if (x==0):
            res.append(1) 
        else:
            res.append(0)
    return res

"""
C'est la fonction u1 appliquée en un seul 
point cette fois.
"""
def u1_bis(X,L):
    u=0
    if (X==0):
        u=1
    return u

"""
C'est l'interpolant de la fonction u1.
Elle retourne une liste.
"""
def p_u1(x,x_m,h):
    n=len(x_m)
    p=[]
    for i in range (0,len(x)):
        s=0
        for l in range (0,n):
            s=s+u1_bis(x_m[l],L)*Sh(h,x[i]-x_m[l])
        p.append(s)
    return p

## On trace d'abord la fonction et ensuite son interpolation.
## Pour plus de lisibilité, on affiche les deux fonctions 
## séparément, mais l'une en dessous de l'autre.

plt.plot(x_m,u1(x_m,L), color="blue")
plt.scatter(x_grille,y_grille,color='red',marker='.',linewidth = 0.1)
plt.title("La fonction u1")
plt.show()

plt.plot(x,p_u1(x,x_m,h))
plt.scatter(x_grille,y_grille,color='red',marker='.',linewidth = 0.1)
plt.title("Interpolation de la fonction u1")
plt.show()

#%%

"""
Cette fonction correpond à la fonction u2 du sujet:
    elle vaut 1 si abs(x)<=L/4, 0 sinon.
Elle renvoie une liste.
"""
def u2(X,L):
    res=[]
    for x in X:
        if (abs(x)<=(L/4)):
            res.append(1)
        else:
            res.append(0)
    return res

"""
C'est la fonction u2 appliquée en un seul 
point cette fois.
"""
def u2_bis(x,L):
    u=0
    if (abs(x)<=(L/4)):
        u=1   
    return u

"""
C'est l'interpolant de la fonction u2.
Elle retourne une liste.
"""
def p_u2(x,x_m,h):
    n=len(x_m)
    p=[]
    for i in range (0,len(x)):
        s=0
        for l in range (0,n):
            s=s+u2_bis(x_m[l],L)*Sh(h,x[i]-x_m[l])
        p.append(s)
    return p

## On trace d'abord la fonction et ensuite son interpolation.
## Pour plus de lisibilité, on affiche les deux fonctions 
## séparément, mais l'une en dessous de l'autre.


plt.plot(x_m,u2(x_m,L), color="blue")
plt.scatter(x_grille,y_grille,color='red',marker='.',linewidth = 0.1)
plt.title("La fonction u2")
plt.show()

plt.plot(x,p_u2(x,x_m,h))
plt.scatter(x_grille,y_grille,color='red',marker='.',linewidth = 0.1)
plt.title("Interpolation de la fonction u2")
plt.show()
    


#%%
"""
Cette fonction correpond à la fonction u3 du sujet:
    - elle vaut 3/(L(x+L/3)) si x appartient à ]-L/3,0]
    - elle vaut 1-3/Lx si x appartient à ]0,L/3]
    - elle vaut 0 dans les autres cas.
Elle renvoie une liste.
"""
def u3(X,L):
    res=[]
    for x in X:
        if (x <=0. ) and (x> (-L/3)):
            res.append(3./(L*(x+L/3)))
        elif (x>0.) and (x <=(L/3)):
            res.append(1-3./(L*x))
        else:
            res.append(0)
    return res


"""
C'est la fonction u3 appliquée en un seul 
point cette fois.
"""
def u3_bis(x,L):
    u=0
    if ((x <=0. ) and (x> (-L/3))):
        u=3./(L*(x+L/3))
    elif (x>0.) and (x <=(L/3)):
        u=1-3./(L*x)
    return u

"""
C'est l'interpolant de la fonction u3.
Elle retourne une liste.
"""
def p_u3(x,x_m,h):
    n=len(x_m)
    p=[]
    for i in range (0,len(x)):
        s=0
        for l in range (0,n):
            s=s+u3_bis(x_m[l],L)*Sh(h,x[i]-x_m[l])
        p.append(s)
    return p


## On trace d'abord la fonction et ensuite son interpolation.
## Pour plus de lisibilité, on affiche les deux fonctions 
## séparément, mais l'une en dessous de l'autre.

plt.plot(x_m,u3(x_m,L), color="blue")
plt.scatter(x_grille,y_grille,color='red',marker='.',linewidth = 0.1)
plt.title("La fonction u3")
plt.show()

plt.plot(x,p_u3(x,x_m,h))
plt.scatter(x_grille,y_grille,color='red',marker='.',linewidth = 0.1)
plt.title("Interpolation de la fonction u3")
plt.show()



#%%
### QUESTION 4 ###

# initialisation
N=24 # nombre de points 
h=2*np.pi/N # le pas

x=np.linspace(0,2*np.pi,int((2*np.pi)/h )) # l'intervalle en fonction du 
# nombre de points et du pas


"""
DerSpecPer est la fonction qui calcule la dérivation
spectrale périodique  d'une fonction périodique.
Elle utilise la matrice de dérivation Dn
Elle renvoie une liste.
"""
def DerSpecPer(v,N,h):
    M=np.zeros((N,N))
    for i in range(1,N+1):
        for j in range (1,N+1):
            if (i==j) or (np.sin(j*h/2)==0):
                M[i-1,j-1]=0
            elif i<j:
                M[i-1,j-1]=0.5*((-1)**(j-i+1))*(np.cos((j-i)*h/2)/np.sin((j-i)*h/2))
    M2=M-np.transpose(M)
    return np.dot(M2,v)


"""
DerSpecPerDFT est la fonction qui calcule la dérivation
spectrale périodique  d'une fonction périodique.
Elle utilise la DFT.
Elle renvoie une liste.
"""
def DerSpecPerDFT(v,N):
    v_chapeau=np.fft.fft(v)/N
    w_chapeau=[]
    k=-N/2+1
    for i in range(0,N-1):
        w_chapeau.append(1j*k*v_chapeau[i])
        k=k+1
    w_chapeau.append(0)
    return -2*np.pi*np.real(np.fft.ifft(w_chapeau))
## NB: on a un problème ici: on a du multipier par -2pi pour obtenir
## le même résultat qu'avec la méthode de la matrice de dérivation.
## On a également vu qu'en ne divisant pas par N et en divisant
## par -2pi, on obtenait également un bon résultat.
## Nous n'avons pas réussi à résoudre ce problème.


"""
C'est la fonction f1 du sujet qui correspond
à 1-max(0,abs(x-pi)/2).
Elle renvoie une liste.
"""
def f1(X):
    p=np.zeros(len(X))
    for i in range (0,len(X)):
        a=abs(X[i]-np.pi)/2
        if (1-a)<=0:
            p[i]=0
        else:
            p[i]=(1-a)
    return p


## Ici, on affiche la fonction f1 ainsi que
## sa dérivée spectrale d'après les deux méthodes
plt.plot(x,f1(x))
plt.title("f1(x)=1-max(0,abs(x-pi)/2)")
plt.show()
plt.plot(x,DerSpecPer(f1(x),N,h))
plt.title("Dérivée spectrale de f1 avec Dn")
plt.show()
plt.plot(x,DerSpecPerDFT(f1(x),N))
plt.title("Dérivée spectrale de f1 avec DFT")
plt.show()

"""
C'est la fonction f2 du sujet qui correspond
à exp(sin(x)).
Elle renvoie une liste.
"""
def f2(X):
    p=np.zeros(len(X))
    for i in range (0,len(X)):
        p[i]=np.exp(np.sin(X[i]))
    return p

## Ici, on affiche la fonction f1 ainsi que
## sa dérivée spectrale d'après les deux méthodes
plt.plot(x,f2(x))
plt.title("f2(x)=exp(sin(x))")
plt.show()
plt.plot(x,DerSpecPer(f2(x),N,h))
plt.title("Dérivée spectrale de f2 avec Dn")
plt.show()
plt.plot(x,DerSpecPerDFT(f2(x),N))
plt.title("Dérivée spectrale de f2 avec DFT")
plt.show()



#%%
### QUESTION 5 ###

# Initialisation

N=128 # nombre de points
h=2*np.pi/N # le pas
dt=h/4 # le delta_t
x=np.linspace(0,2*np.pi,N) # l'intervalle étudié
T=8 #le temps maximal

"""
La fonction c est la fonction correspondant
au terme dans l'équation des ondes.
Elle correspond à 1/5+sin^2(x-1)
Elle est appliquée point par point.
"""
def c (x):
    return 0.2+(np.sin(x-1))**2

"""
La fonction u0 est l'initialisation
de l'équation des ondes.
Elle correspond à exp(-100(x-1)^2)
Elle est appliquée point par point.
"""
def u0 (x):
    return np.exp(-100*(x-1)**2)


"""
v_moins1 sert à obtenir le terme v^(-1)
qui est nécessaire dans lé résolution
du problème de l'équation des ondes.
On définit v^(-1) par la fonction
exp(-100(x-0.2*dt-1)^2).
Elle est appliquée point par point.
"""
def v_moins1(x,dt):
    return np.exp(-100*(x-0.2*dt-1)**2)

"""
La fonction resol permet de résoudre le schéma 
numérique. Nous avons ici fait le choix
d'utiliser la fonction DerSpecPer et de prendre
le v^(-1) comme initialisation.
Elle retourne une matrice.
"""
def resol (x,N,h,dt,T):
    vav=v_moins1(x,dt)
    v=u0(x)
    nbetape=int(T/dt)
    M=np.zeros((len(x),nbetape))
    M[:,0]=vav
    M[:,1]=v
    for i in range (2,nbetape):
        Q=DerSpecPer(v,N,h)
        vap=vav-2*dt*c(x)*Q
        vav=v
        v=vap
        M[:,i]=v
    return M



## On affecte la fonction resol à une variable.
## En effet, il est plus simple de calculer
## et d'afficher la figure séparément car 
## la fonction resol prend du temps à s'effectuer.
## C'est pour cela que l'on a fait deux cellules pour
## cette question
reso=resol(x,N,h,dt,T)

#%% Question 5

t=np.linspace(0,T,len(reso[1,:])) #on définit le temps de 0 a 8 avec un pas de dt

XX, TT= np.meshgrid(x,t) # on définit les points de la grille sur X,Y

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')      #Graphe 3D

for i in range(len(TT)):          #Trace toutes les courbes en 3D
    plt.plot(x,TT[i],reso[:,i],color="black",linewidth=0.25)

plt.title("Propagation de l'onde")
ax.set_zlim3d(0,5)     # limite de l'axe z
ax.view_init(30, -80) #Orientation du graphe 3D
plt.ylabel("temps")
plt.xlabel("x")
plt.show()

                    
#%%

### QUESTION 6 ###


maxN=50 # nombre de points max pris

"""
Correspond à la fonction abs(sin(x))^3
"""
def g1(x):
    return abs(np.sin(x))**3

"""
Correspond à la dérivée de g1
"""
def g1der(x):
    return 3*np.sin(x)*np.cos(x)*abs(np.sin(x))

"""
Correspond à la fonction exp(-sin(x/2)^(-2))
"""
def g2(x):
    return np.exp(-np.sin(x/2)**(-2))

"""
Correspond à la dérivée de g2
"""
def g2der(x):
    return np.exp(-np.sin(x/2)**(-2))*np.cos(x/2)/(np.sin(x/2)**(-3))

"""
Correspond à la fonction 1/(1+sin^2(x/2))
"""
def g3(x):
    return 1/(1+np.sin(x/2)**2)

"""
Correspond à la dérivée de g3
"""
def g3der(x):
    return -np.sin(x/2)*np.cos(x/2)*(1/(1+np.sin(x/2)**2))**2

"""
Correspond à la fonction sin(10x)
"""
def g4(x):
    return np.sin(10*x)

"""
Correspond à la dérivée de g4
"""
def g4der(x):
    return 10*np.cos(10*x)



## On crée 4 listes d'erreurs, de 1 à 4,
## et on les remplit par la norme infinie
## de la différence de la dérivée spectrale
## de la fonction (en utilisant DerSpecPer,
## c'est à dire la matrice de dérivation) et
## de la dérivée de la fonction.
E1=[]
E2=[]
E3=[]
E4=[]
for N in range (6,maxN,2):
    x=np.linspace(0,2*np.pi,N)
    h=2*np.pi/N
    E1.append(LA.norm(DerSpecPer(g1(x),N,h)-g1der(x),np.inf))
    E2.append(LA.norm(DerSpecPer(g2(x),N,h)-g2der(x),np.inf))
    E3.append(LA.norm(DerSpecPer(g3(x),N,h)-g3der(x),np.inf))
    E4.append(LA.norm(DerSpecPer(g4(x),N,h)-g4der(x),np.inf))


x=np.linspace(6,maxN,(maxN-6)/2)  
plt.plot(x,E1, ".")
plt.yscale("log")
plt.ylim(10**(-5),10**2)
plt.title("Erreur de |sin(x)|^3")
plt.show()
    
plt.plot(x,E2, ".")
plt.yscale("log")
plt.ylim(10**(-5),10**2)
plt.title("Erreur de exp(-sin(x/2)^(-2))")
plt.show()

plt.plot(x,E3, ".")
plt.yscale("log")
plt.ylim(10**(-5),10**2)
plt.title("Erreur de 1/(1+sin^2(x/2))")
plt.show()  

plt.plot(x,E4, ".")
plt.yscale("log")
plt.ylim(10**(-5),10**2)
plt.title("Erreur de sin(10x)")
plt.show()





























