##---------------------------------------##
##------------- Gazzo Sandro ------------##
##------------- Dellouve Théo -----------##
##---------------------------------------##

#importation des librairies

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
#%%




def GaussTcheb(N) :
    """
    Cette fonction calcule les N+1 points de collocation de Gauss
    """
    gauss=np.zeros(N+1)
    poids=np.ones(N+1)
    for i in range(0,N+1):
        gauss[i]=np.cos((i+0.5)*np.pi/(N+1))
    return gauss, poids*np.pi/(N+1)


def GaussLobattoTcheb(N):
    """
    Cette fonction calcule les N+1 points de collocation de Gauss-Lobatto
    """
    lob=np.zeros(N+1)
    poids=np.zeros(N+1)
    for i in range(0,N+1):
        lob[i]=np.cos(np.pi*i/N)
        if (i==0) or (i==N):
            poids[i]=np.pi/(2*N)
        else:
            poids[i]=np.pi/N
    return lob, poids



def norme_carree_N(n,N,choix):
    """
    Cette fonction calcule la norme N au carré.
    - n correspond à l'indice de c_n
    - N correspond au nombre de points choisis
    - choix=1: Point de collocation de Gauss
    - choix=2: Point de collocation de Gauss-Lobatto
    """
    assert(choix==1 or choix==2), "Le choix est soit 1 soit 2."
    if (choix==1):
        if (n==0):
            return np.pi
        else:
            return np.pi/2
    if (choix==2):
        if ((n==0) or (n==N)):
            return np.pi
        else:
            return np.pi/2
        
def T(n,x):
    """
    Permet le calcul du polynôme de Tchebyshev T_n.
    - n est le degré que l'on veut atteindre
    - x est le vecteur 
    """
    N=np.size(x)
    T_sec=np.ones(N)
    T_pre=x
    if n==0 :
        return T_sec
    if n==1:
        return T_pre
    else:
        for i in range (2,n+1):
            T_act=np.zeros(N)
            for j in range (0,N):
                T_act[j]=2*x[j]*T_pre[j]-T_sec[j]
            T_sec=T_pre
            T_act1=T_act
            T_pre=T_act1
        return T_act
        

def f_tilde(f,choix):
    """
    Cette fonction calcule les coefficients de l'interpolant de Tchebyshev.
    - f correpond au vecteur que l'on veut interpoler. 
      Sa longeur est strict. supérieure à 1.
    - choix=1: Point de collocation de Gauss
    - choix=2: Point de collocation de Gauss-Lobatto
    """
    assert(choix==1 or choix==2), "Le choix est soit 1 soit 2."
    assert(np.size(f)!=1), "On va diviser par zéro: impossible."
    N=np.size(f)-1
    f_t=np.zeros(N+1) 
    if (choix==1):
        gauss=GaussTcheb(N)
        x=gauss[0]
        w=gauss[1]
    elif (choix==2):
        lob=GaussLobattoTcheb(N)
        x=lob[0]
        w=lob[1]    
    for n in range (1,N):
        T_act=T(n,x)
        for j in range (0,N+1):
            f_t[n]=f_t[n]+f[j]*T_act[j]*w[j]
        f_t[n]=f_t[n]/norme_carree_N(n,N+1,choix)
    T_act=T(0,x)
    for j in range (0,N+1):
            f_t[0]=f_t[0]+f[j]*T_act[j]*w[j]
    f_t[0]=f_t[0]/norme_carree_N(0,N,choix)
    T_act=T(N,x)
    for j in range (0,N+1):
            f_t[N]=f_t[N]+f[j]*T_act[j]*w[j]
    f_t[N]=f_t[N]/norme_carree_N(N,N,choix)
    return f_t
    

    

def CoeffDeriveeTcheb(f):
    """
    Cette fonction nous permet de calculer les coefficients de la dérivée de 
    Tchebyshev de f.
    - f est la fonction à dériver
    """
    N=np.size(f)-1
    c=np.ones(N+1)
    c[0]=2
    c[N]=2
    v=np.zeros(N+2)
    for k in range(N-1,-1,-1):
        v[k]=(v[k+2]+2*(k+1)*f[k+1])/c[k]
    derivee=v[0:N+1]
    return derivee


def DeriveeTcheb(f,x,choix):
    """
    Cette fonction calcule, en utilisant une méthode spectrale de Tchebyshev, 
    la dérivée aux points x d'une fonction données par les valeurs de f dans 
    les noeuds de Tchebyshev.
    - f correpond à la fonction à dériver
    - x correspond aux noeuds de Tchebyshev
    - choix=1: Point de collocation de Gauss
    - choix=2: Point de collocation de Gauss-Lobatto
    """
    assert(choix==1 or choix==2), "Le choix est soit 1 soit 2."
    assert(np.size(f)!=1), "On va diviser par zéro: impossible."
    ft=f_tilde(f,choix)
    N=np.size(f)-1
    coeff=CoeffDeriveeTcheb(ft)
    s=np.zeros(len(x))
    for k in range(0,N+1):
        T_act=T(k,x)
        s=s+coeff[k]*T_act
    return s


N=25

## ----------------------------------------------------------
##### GAUSS #####
## y=1
x=np.ones(N)
x_test=np.linspace(-1,1,100)
test1=np.zeros(100)
test1=DeriveeTcheb(x,x_test,1)
plt.plot(x_test,test1,label="Approchée")
plt.plot(x_test,np.zeros(len(x_test)),label="Exacte")
plt.ylim(-1,1)
plt.legend()
plt.title("Dérivée de la fonction f(x)=1 par les points de \n collocation de Gauss et par la vraie solution")
plt.show()

## y=x
x=GaussTcheb(N)[0]
x_test=np.linspace(-1,1,100)
test1=np.zeros(100)
test1=DeriveeTcheb(x,x_test,1)
plt.plot(x_test,test1,label="Approchée")
plt.plot(x_test,np.ones(len(x_test)),label="Exacte")
plt.legend()
plt.ylim(0,2)
plt.title("Dérivée de la fonction f(x)=x par les points de \n collocation de Gauss et par la vraie solution")
plt.show()

## y=x**2
x=GaussTcheb(N)[0]**2
x_test=np.linspace(-1,1,100)
test1=np.zeros(100)
test1=DeriveeTcheb(x,x_test,1)
plt.plot(x_test,test1,label="Approchée")
plt.plot(x_test,2*x_test,label="Exacte")
plt.legend()
plt.title("Dérivée de la fonction f(x)=x^2 par les points de \n collocation de Gauss et par la vraie solution")
plt.show()


## y=x**3
x=GaussTcheb(N)[0]**3
x_test=np.linspace(-1,1,100)
test1=np.zeros(100)
test1=DeriveeTcheb(x,x_test,1)
plt.plot(x_test,test1,label="Approchée")
plt.plot(x_test,3*(x_test**2),label="Exacte")
plt.legend()
plt.title("Dérivée de la fonction f(x)=x^3 par les points de \n collocation de Gauss et par la vraie solution")
plt.show()


## y=abs(x)
x=abs(GaussTcheb(N)[0])
x_test=np.linspace(-1,1,100)
test1=np.zeros(100)
test1=DeriveeTcheb(x,x_test,1)
plt.plot(x_test,test1,label="Approchée")
plt.plot(x_test,x_test/abs(x_test),label="Exacte")
plt.legend()
plt.title("Dérivée de la fonction f(x)=|x| par les points de \n collocation de Gauss et par la vraie solution")
plt.show()

## y=exp(x)
x=np.exp(GaussTcheb(N)[0])
x_test=np.linspace(-1,1,100)
test1=np.zeros(100)
test1=DeriveeTcheb(x,x_test,1)
plt.plot(x_test,test1,label="Approchée")
plt.plot(x_test,np.exp(x_test),label="Exacte")
plt.legend()
plt.title("Dérivée de la fonction f(x)=e^x par les points de \n collocation de Gauss et par la vraie solution")
plt.show()

## y=cos(pi*x)
x=np.cos(GaussLobattoTcheb(N)[0]*np.pi)
x_test=np.linspace(-1,1,100)
test1=np.zeros(100)
test1=DeriveeTcheb(x,x_test,2)
plt.plot(x_test,test1,label="Approchée")
plt.plot(x_test,-np.pi*np.sin(np.pi*x_test),label="Exacte")
plt.legend()
plt.title("Dérivée de la fonction f(x)=cos(pi*x) par les points de \n collocation de Gauss et par la vraie solution")
plt.show()


## y=1/x^2
x=1/(GaussLobattoTcheb(N)[0]**2)
x_test=np.linspace(-1,1,100)
test1=np.zeros(100)
test1=DeriveeTcheb(x,x_test,2)
plt.plot(x_test,test1,label="Apprcohée")
plt.plot(x_test,-2/(x_test**3),label="Exacte")
plt.ylim(-2500,2500)
plt.legend()
plt.title("Dérivée de la fonction f(x)=1/x^2 par les points de \n collocation de Gauss et par la vraie solution")
plt.show()


## -----------------------------------------------------
#### Gauss-Lobatto ####
## y=1
x=np.ones(N)
x_test=np.linspace(-1,1,100)
test1=np.zeros(100)
test1=DeriveeTcheb(x,x_test,2)
plt.plot(x_test,test1,label="Approchée")
plt.plot(x_test,np.zeros(len(x_test)),label="Exacte")
plt.ylim(-1,1)
plt.legend()
plt.title("Dérivée de la fonction f(x)=1 par les points de \n collocation de Gauss-Lobatto et par la vraie solution")
plt.show()

## y=x
x=GaussLobattoTcheb(N)[0]
x_test=np.linspace(-1,1,100)
test1=np.zeros(100)
test1=DeriveeTcheb(x,x_test,2)
plt.plot(x_test,test1,label="Approchée")
plt.plot(x_test,np.ones(len(x_test)),label="Exacte")
plt.legend()
plt.ylim(0,2)
plt.title("Dérivée de la fonction f(x)=x par les points de \n collocation de Gauss-Lobatto et par la vraie solution")
plt.show()

## y=x**2
x=GaussLobattoTcheb(N)[0]**2
x_test=np.linspace(-1,1,100)
test1=np.zeros(100)
test1=DeriveeTcheb(x,x_test,2)
plt.plot(x_test,test1,label="Approchée")
plt.plot(x_test,2*x_test,label="Exacte")
plt.legend()
plt.title("Dérivée de la fonction f(x)=x^2 par les points de \n collocation de Gauss-Lobatto et par la vraie solution")
plt.show()


## y=x**3
x=GaussLobattoTcheb(N)[0]**3
x_test=np.linspace(-1,1,100)
test1=np.zeros(100)
test1=DeriveeTcheb(x,x_test,2)
plt.plot(x_test,test1,label="Approchée")
plt.plot(x_test,3*(x_test**2),label="Exacte")
plt.legend()
plt.title("Dérivée de la fonction f(x)=x^3 par les points de \n collocation de Gauss-Lobatto et par la vraie solution")
plt.show()


## y=abs(x)
x=abs(GaussLobattoTcheb(N)[0])
x_test=np.linspace(-1,1,100)
test1=np.zeros(100)
test1=DeriveeTcheb(x,x_test,2)
plt.plot(x_test,test1,label="Approchée")
plt.plot(x_test,x_test/abs(x_test),label="Exacte")
plt.legend()
plt.title("Dérivée de la fonction f(x)=|x| par les points de \n collocation de Gauss-Lobatto et par la vraie solution")
plt.show()

## y=exp(x)
x=np.exp(GaussLobattoTcheb(N)[0])
x_test=np.linspace(-1,1,100)
test1=np.zeros(100)
test1=DeriveeTcheb(x,x_test,2)
plt.plot(x_test,test1,label="Approchée")
plt.plot(x_test,np.exp(x_test),label="Exacte")
plt.legend()
plt.title("Dérivée de la fonction f(x)=e^x par les points de \n collocation de Gauss-Lobatto et par la vraie solution")
plt.show()

## y=cos(pi*x)
x=np.cos(GaussLobattoTcheb(N)[0]*np.pi)
x_test=np.linspace(-1,1,100)
test1=np.zeros(100)
test1=DeriveeTcheb(x,x_test,2)
plt.plot(x_test,test1,label="Approchée")
plt.plot(x_test,-np.pi*np.sin(np.pi*x_test),label="Exacte")
plt.title("Dérivée de la fonction f(x)=cos(pi*x) par les points de \n collocation de Gauss-Lobatto et par la vraie solution")
plt.legend()
plt.show()

## y=1/x^2
x=1/(GaussLobattoTcheb(N)[0]**2)
x_test=np.linspace(-1,1,100)
test1=np.zeros(100)
test1=DeriveeTcheb(x,x_test,2)
plt.plot(x_test,test1,label="Apprcohée")
plt.plot(x_test,-2/(x_test**3),label="Exacte")
plt.ylim(-2500,2500)
plt.legend()
plt.title("Dérivée de la fonction f(x)=1/x^2 par les points de \n collocation de Gauss-Lobatto et par la vraie solution")
plt.show()

## --------------------------------------------------------------

#%%
### Exo2

def DCT(f):
    """
    Cette fonction calcule la Transformation en Cosinus Discrète (TCD) de f.
    """
    assert(np.size(f)!=1), "On va diviser par zéro: impossible."
    N=np.size(f)-1
    f_t=np.zeros(N+1)
    for k in range (0,N+1):
        for j in range (0,N+1):
            if ((j==0) or (j==N)):
                f_t[k]=f_t[k]+(f[j]/2)*np.cos(j*k*np.pi/N)
            else:
                f_t[k]=f_t[k]+f[j]*np.cos(j*k*np.pi/N)
        if ((k==0) or (k==N)):
            f_t[k]=f_t[k]/N
        else:
            f_t[k]=2*f_t[k]/N
    return f_t
    




def hyperbol(x,t,u):
    """
    Cette fonction calcule une approximation de la fonction u(x,t) 
    à un point de temps t.
    - x corrsespond au points de collocation de Gauss-Lobatto
    - t correpond au temps actuel
    - u correpond à la fonction u(x,t)
    """
    assert(np.size(x)!=1), "On va diviser par zéro: impossible."
    N=np.size(x)-1
    
    a=DCT(u)
    
    S=np.zeros(N+2)
    for n in range(N-1,0,-1):
        S[n]=S[n+2]+(n+1)*a[n+1]
    S[0]=(S[2]+a[1])/2
    a1=2*S[0:N+1]
    
    hhh=np.zeros(N+1)
    for j in range (0,N):
        somme=0
        for n in range(0,N+1):
            somme=somme+a1[n]*np.cos(np.pi*j*n/N)
        hhh[j]=np.exp(u[j]+np.cos(np.pi*j/N))*somme
        
    return f(x,t)-hhh

#def RungeKutta(x,K,T):
#    h = T/K
#    Y1=u_tcheb(x,0)
#    for n in range(K):
#        t=n*h
#        RK1 = hyperbol(x,t,Y1)
#        RK2 = hyperbol(x,t+h*RK1/2,Y1)
#        RK3 = hyperbol(x,t+h*RK2/2,Y1)
#        RK4 = hyperbol(x,t+h*RK3,Y1)
#        Y1 = Y1+h*(RK1+2*RK2+2*RK3+RK4)/6
#        Y1[-1]=0
#    return Y1
#    
#
#
#def EulerAmeliore(f,T,K,U0):
#    U=np.zeros((np.size(U0),K+1))
#    U[:,0]=U0
#    h=T/K
#    for i in range(1,K+1):
#        U[:,i]=U[:,i-1]+h*(hyperbol(f,U[:,i-1])+hyperbol(f,U[:,i-1])+h*hyperbol(f,U[:,i-1]))/2
#    return U

def u_tcheb(x,t):
    """
    Calcule u(x,t)=(x+1)*t
    - x est la coordonée de l'espace (en vecteur)
    - t est la coordonée en temps (en unité)
    """
    return (x+1)*t

def f(x,t):
    """
     Calcule f(x,t)=x+1+exp(u(x,t)+x)*t
    - x est la coordonée de l'espace (en vecteur)
    - t est la coordonée en temps (en unité)
    """
    return x+1+np.exp(u_tcheb(x,t) + x)*t


#def Euler_modifie(x,t,delta_t,Y1):
#    # Fonction qui progresse d'un pas de temps par la méthode d'Euler modifié
#    
#    Y2 = Y1+delta_t/2*hyperbol(x,t,Y1)
#    resultat = Y1 + delta_t*hyperbol(x,t+delta_t/2,Y2)
#    
#    resultat[-1]=0
#    
#    return resultat
    
#def resolution(x,K,T):
#    # Fonction qui calcule la solution exacte de notre problème en progressant
#    # en temps avec la méthode d'Euler modifié avant comme condition initiale u(x,0).
#    
#    delta_t = T/K
#    resultat=u_tcheb(x,0)
#    for j in range(1,K+1):
#        tj = j*delta_t
#        resultat = Euler_modifie(x,tj,delta_t,resultat)
#    return resultat

def resolutionEulerModifie(x,K,T):
    """
    Permet de résoudre, en utilisant la méthode de Euler Modifié sur un 
    intervalle de temps T et par un pas h, le problème hyperbolique posé
    avec la fonction hyperbol(x,t,u)
    - x correspond aux points de collocation de Gauss-Lobatto
    - K correpond au nombre d'itérations de temps à effectuer
    - T correspond à l'intervalle de temps
    """
    h = T/K
    Y1=u_tcheb(x,0)
    for j in range(1,K+1):
        tj = j*h
        Y2 = Y1+h/2*hyperbol(x,tj,Y1)
        Y1 = Y1 + h*hyperbol(x,tj+h/2,Y2)
    
        Y1[-1]=0 # on satisfait la condition au bord
    return Y1


## -------------------------------------------------------
N=25
Tps=[0.3,0.5,0.9,1]
K=1000
x=GaussLobattoTcheb(N)[0]
x2=np.arange(-1,1,2/200) # pour pouvoir calculer la solution exacte
for t in Tps:
    #test=RungeKutta(x,K,t)
    test=resolutionEulerModifie(x,K,t)
    
    plt.plot(x,test,label="Approchée")
    
    plt.plot(x2,u_tcheb(x2,t),"--",label="Exacte")
    plt.title("Comparaison entre la solution exacte u(x,t)=(x+1)*t \n et la solution approchée pour N=25 et T="+str(t))
    plt.legend()
    plt.show()


#%%
### Exo 3
def g(t):
    """
    Calcule la fonction g(t)=-sin(5pi*t) où t est une unité de temps.
    """
    return -np.sin(5*np.pi*t)


def u_colloc(x,t):
    """
    Calcule la fonction u(x,t)=g(t-x-1) si x <= t-1, 0 sinon.
    - x est la coordonée de l'espace (en vecteur)
    - t est la coordonée en temps (en unité)
    """
    res=np.zeros(len(x))
    for i in range (0,len(x)):
        if (x[i]<=(t-1)):
            res[i]=g(t-x[i]-1)
        else:
            res[i]=0
    return res
      




def resol_pbTcheb(x,u):
    """
    Cette fonction calcule une approximation de la fonction u(x,t) 
    à un point de temps t par la méthode de collocation de Tchebyshev.
    - x corrsespond au points de collocation de Gauss-Lobatto
    - u correpond à la fonction u(x,t)
    
    Attention: dans notre problème, f(x,t)=0, la fonction ne dépend donc
    pas du temps.
    """
    assert(np.size(x)!=1), "On va diviser par zéro: impossible."
    N=np.size(x)-1
    
    a=DCT(u)
    
    S=np.zeros(N+2)
    for n in range(N-1,0,-1):
        S[n]=S[n+2]+(n+1)*a[n+1]
    S[0]=(S[2]+a[1])/2
    a1=2*S[0:N+1]
    
    
    du=np.zeros(N+1)
    for j in range (0,N):
        for n in range(0,N+1):
            du[j]=du[j]+a1[n]*np.cos(np.pi*j*n/N)
        
    return -du


#def RungeKuttaColloc(x,K,T):
#    h = T/K
#    Y1=u2(x,0)
#    for n in range(K):
#        t=n*h
#        RK1 = resol_pb(x,Y1)
#        RK2 = resol_pb(x,Y1)
#        RK3 = resol_pb(x,Y1)
#        RK4 = resol_pb(x,Y1)
#        Y1 = Y1+h*(RK1+2*RK2+2*RK3+RK4)/6
#        Y1[-1]=g(t)
#    return Y1

#def Euler_modifie2(x,t,delta_t,Y1):
#    # Fonction qui progresse d'un pas de temps par la méthode d'Euler modifié
#    
#    Y2 = Y1+delta_t/2*resol_pb(x,Y1)
#    resultat = Y1 + delta_t*resol_pb(x,Y2)
#    
#    resultat[-1]=g(t)
#    
#    return(resultat)
#    
#def CollocTchebTransport(x,K,T):
#    # Fonction qui calcule la solution exacte de notre problème en progressant
#    # en temps avec la méthode d'Euler modifié avant comme condition initiale u(x,0).
#    
#    delta_t = T/K
#    resultat=u_colloc(x,0)
#    
#    for j in range(1,K+1):
#        tj = j*delta_t
#        resultat = Euler_modifie2(x,tj,delta_t,resultat)
#    return(resultat)



def CollocTchebTransport(x,K,T):
    """
    Permet de résoudre, en utilisant la méthode de Euler Modifié sur un 
    intervalle de temps T et par un pas h, le problème hyperbolique posé
    avec la fonction hyperbol(x,t,u)
    - x correspond aux points de collocation de Gauss-Lobatto
    - K correpond au nombre d'itérations de temps à effectuer
    - T correspond à l'intervalle de temps
    """
    h = T/K
    Y1=u_colloc(x,0)
    for j in range(1,K+1):
        tj = j*h
        Y2 = Y1+h/2*resol_pbTcheb(x,Y1)
        Y1 = Y1 + h*resol_pbTcheb(x,Y2)
        
        Y1[-1]=g(tj) # on satisfait la condition au bord
    return Y1

#--------------------------------------------------------------
K=1000
Tps=[1,1.4,4,4.2,4.4]
N=[14,18,22,26,30,34]
x2=np.arange(-1,1,2/200) # pour pouvoir tracer la solution exacte
stock=np.zeros(len(N))
i=0
for le_N in N:
    x=GaussLobattoTcheb(le_N)[0]
    for t in Tps:
        print(t)
        test=CollocTchebTransport(x,K,t)
        plt.plot(x,test,"o")
        plt.plot(x2,u_colloc(x2,t))
        plt.title("Comparaison entre la solution exacte et les points de collocation \n de Gauss-Lobatto la solution approchée pour N="+str(le_N)+" et T="+str(t))
        plt.show()
    stock[i]=LA.norm(test-u_colloc(x,t))
    i=i+1

# on trace l'erreur 
plt.plot(N,stock,"o")
plt.yscale("log")
plt.title("erreur en t=4.4")
plt.show()



#%%
### Galerkin : ne marche pas ####

def M(taille):
    N=taille+1
    D=np.zeros((N,N))
    for i in range (0,N):
        for j in range (0,N):
            D[i,j]=((-1)**((i+1)+(j+1)))*np.pi
            if (i==j):
                D[i,j]=D[i,j]+ np.pi/2
    return D


def S(taille):
    N=taille+1
    D=np.zeros((N,N))
    for n in range (0,N):
        for m in range (0,N):
            if (((n+1)%2==1) and ((m>n) or ((m+1)%2==1))):
                D[n,m]=np.pi*(m+1)
            elif (((m+1)%2==1) and ((n+1)%2==0) and (m>n)):
                D[n,m]=-np.pi*(m+1)
    return D        
    

def resol_pbGalerkin(x,u):
    """
    Cette fonction calcule une approximation de la fonction u(x,t) 
    à un point de temps t par la méthode de collocation de Galerkin.
    - x corrsespond au points de collocation de Gauss-Lobatto
    - u correpond à la fonction u(x,t)
    
    Attention: dans notre problème, f(x,t)=0, la fonction ne dépend donc
    pas du temps.
    """
    assert(np.size(x)!=1), "On va diviser par zéro: impossible."
    N=np.size(x)-1
    
    a=DCT(u)
    
    a1=np.dot(LA.inv(M(N)),np.dot(S(N),a))
    
    
    du=np.zeros(N+1)
    for j in range (0,N):
        for n in range(0,N+1):
            du[j]=du[j]+a1[n]*np.cos(np.pi*j*n/N)
        
    return -du



def CollocGalerkinTransport(x,K,T):
    """
    Permet de résoudre, en utilisant la méthode de Euler Modifié sur un 
    intervalle de temps T et par un pas h, le problème hyperbolique posé
    avec la fonction hyperbol(x,t,u)
    - x correspond aux points de collocation de Gauss-Lobatto
    - K correpond au nombre d'itérations de temps à effectuer
    - T correspond à l'intervalle de temps
    """
    h = T/K
    Y1=u_colloc(x,0)
    for j in range(1,K+1):
        tj = j*h
        Y2 = Y1+h/2*resol_pbGalerkin(x,Y1)
        Y1 = Y1 + h*resol_pbGalerkin(x,Y2)
        
        Y1[-1]=g(tj) # on satisfait la condition au bord
    return Y1

#--------------------------------------------------------------
K=1000
Tps=[1,1.4,4,4.2,4.4]
N=[14,18,22,26,30,34]
x2=np.arange(-1,1,2/200) # pour pouvoir tracer la solution exacte
stock=np.zeros(len(N))
i=0
for le_N in N:
    x=GaussLobattoTcheb(le_N)[0]
    for t in Tps:
        print(t)
        test=CollocGalerkinTransport(x,K,t)
        plt.plot(x,test,"o")
        plt.plot(x2,u_colloc(x2,t))
        plt.title("Comparaison entre la solution exacte et les points de collocation \n de Gauss-Lobatto la solution approchée pour N="+str(le_N)+" et T="+str(t)+"\n")
        plt.show()
    stock[i]=LA.norm(test-u_colloc(x,t))
    i=i+1

# on trace l'erreur 
plt.plot(N,stock,"o")
plt.yscale("log")
plt.title("erreur en t=4.4")
plt.show()























