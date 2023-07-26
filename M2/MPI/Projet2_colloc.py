##---------------------------------------##
##------------- Gazzo Sandro ------------##
##------------- Dellouve Théo -----------##
##---------------------------------------##

#importation des librairies

import numpy as np
import matplotlib.pyplot as plt
from mpmath import *
mp.dps=25; mp.pretty=True
#%%

# Question 1

def CollocFourierDerTemps(phi,N,nu):
    # G(phi) pour la résolution du système différentiel phi' = G(phi)
    # à partir de N points. nu est le parametre de l'équation
    M=np.zeros((N,N)) # matrice de stockage
    h=2*np.pi/N # le pas
    #on construit la matrice de dérivation D
    for i in range(1,N+1):
        for j in range (1,N+1):
            if (i==j) or (np.sin(j*h/2)==0):
                M[i-1,j-1]=0
            elif i<j:
                M[i-1,j-1]=0.5*((-1)**(j-i+1))*(np.cos((j-i)*h/2)/np.sin((j-i)*h/2))
    D=M-np.transpose(M)
    # on calcul G
    F=nu*np.dot(D,phi)
    G=np.dot(D,(F-phi))
    return G

# Question 2
    
def CollocFourierRK3Pas(dt,phi,nu) :
    # execute 1 pas d'une
    # méthode à un pas stable pour la résolution
    # d’un système d’équations différentielles ordinaires d’ordre 1
    # (Runge-Kutta) dans le cas Collocation Fourier
    # pour passer de t=t_n a t=t_{n+1}=t_n+dt.
    # phi est la solution initiale
    # dt est le pas en temps
    # nu est le parametre de l'équation
    N=np.size(phi) #nombre de points
    #algorithme de RK3 :
    U=phi
    G=CollocFourierDerTemps(U,N,nu)
    U=U+(1./3)*dt*G
    G=-(5./9)*G+CollocFourierDerTemps(U,N,nu)
    U=U+(15./16)*dt*G
    G=-(153./128)*G+CollocFourierDerTemps(U,N,nu)
    return U+(8./15)*dt*G
    

# Question 3
    

def CollocFourier(phi,N,nu,dt,T):
    # applique la méthode de collocation de Fourier pour résoudre
    # notre equation dans l’intervalle [0, T] avec un pas dt et
    # une approximation spectrale d’ordre N
    # avec phi la solution initiale
    # nu le parametre de l'équation
    # dt le pas en temps
    pas_tps=int(T/dt) #nombre de pas en temps a faire
    #on fait pas_tps fois un pas pour arriver a T :
    for j in range (0,pas_tps):
        phi=CollocFourierRK3Pas(dt,phi,nu)
    return phi

# Question 4
    
def phi0(x):
    # renvoie la solution initiale sur x
    return 3./(5-4*np.cos(x))

def phi0_chap(k):
    # coefficients de Fourier de la condition initiale phi0(x)
    # à l'odre k
    return 2**(-abs(k))

def phi(x,t,nu):
    # solution exacte en t sur les points de x
    # avec nu le parametre de l'equation
    M=1000 #troncature de la serie de fourier à l'ordre k=1000
    s=0 #initialisation
    #calcul de la somme:
    for k in range (-M,M):
        s=s+(2**(-abs(k))*np.exp(1j*k*(x-t)-nu*(k**2)*t))
    return s
    


def Sh(h,x):
    #h le pas
    # calcul en x le sinus cardinal
    # utilisé dans l'interpolation
    if x==0:
        return 1
    else:
        return (np.sin(np.pi*x/h)/(np.pi*x/h))


def interp_phi(phi0,x,x_j,h):
    # renvoie l'interpolation de phi0 en x à partir
    # des points en x_m qui ont un pas de h
    N=len(x_j) #nombre d points
    p=[] # stockage
    #calcul de l'interpolation
    for i in range (0,len(x)):
        s=0
        for j in range (0,N):
            if x[i]==x_j[j]:
                s=s+phi0[j]
            else:
                s=s+phi0[j]*(1/N)*np.sin(N*(x[i]-x_j[j])/2)*cot((x[i]-x_j[j])/2)
        p.append(s)
    return p
    


#donnees:
dt=1.25*(10**(-3)) # pas en temps
M=100 
N=16 # nombre de points
nu=0.2 # parametre de l'equation
#%% INTERPOLATION
#cas t=0
h=np.pi*2/N # pas en espace
x_m=np.linspace(0,2*np.pi-h,N) # points par discrétisation en N points
x=np.linspace(0,2*np.pi-h,M) #discrétisation en M points
phi_ini=phi0(x_m) #solution initiale
inter=interp_phi(phi_ini,x,x_m,h) #interpolation

#on trace l'interpolation
plt.plot(x_m,phi_ini,'o',color='red', label="solution approchée")
plt.plot(x,inter, label="interpolation")
plt.xlabel("x")
plt.title("Interpolation à t=0")
plt.legend()
plt.show()

#%% INTERPOLATION
#cas t=1
T=1
phi2=CollocFourier(phi_ini,N,nu,dt,T) # approximation en T=1
inter2=interp_phi(phi2,x,x_m,h) #interpolation

#on trace l'interpolation
plt.plot(x,inter2, label="interpolation")
plt.plot(x_m,phi2,'o',color='red',label="solution approchée")
plt.xlabel("x")
plt.title("Interpolation à t=1")
plt.legend()
plt.show()

#%% INTERPOLATION
#cas t=2
T=2
phi3=CollocFourier(phi_ini,N,nu,dt,T)  # approximation en T=2
inter3=interp_phi(phi3,x,x_m,h) #interpolation

#on trace l'interpolation
plt.plot(x,inter3, label="interpolation")
plt.plot(x_m,phi3,'o',color='red',label="solution approchée")
plt.xlabel("x")
plt.title("Interpolation à t=2")
plt.legend()
plt.show()


#%% INTERPOLATION

#on trace l'evolution de l'approximation (les points seulements)
plt.plot(x,inter, label="t=0")
plt.plot(x,inter2, label="t=1")
plt.plot(x,inter3, label="t=2")
plt.xlabel("x")
plt.title("Approximations par interpolation")
plt.legend()
plt.show()

#calcul des solutions exact en T=0, 1 et 2
t=0
sol_ex0=phi(x,t,nu).real
t=1
sol_ex1=phi(x,t,nu).real
t=2
sol_ex2=phi(x,t,nu).real

#on trace les solution exact pour comparer
plt.plot(x,sol_ex0, label="t=0")
plt.plot(x,sol_ex1, label="t=1")
plt.plot(x,sol_ex2, label="t=2")
plt.xlabel("x")
plt.title("Solutions exactes")
plt.legend()
plt.show()

#%% Solutions exactes individuelles
plt.plot(x,sol_ex0)
plt.xlabel("x")
plt.title("Solution exacte à t=0")
plt.show()

plt.plot(x,sol_ex1)
plt.xlabel("x")
plt.title("Solution exacte à t=1")
plt.show()

plt.plot(x,sol_ex2)
plt.xlabel("x")
plt.title("Solution exacte à t=2")
plt.show()
#%% PROJECTION

def projection(N,M):
    # projection de l'approximation en M points a partir de N points (en t=0)
    x=np.arange(0,2*np.pi,2*np.pi/M) # points de sortis
    result=np.zeros(M) # stockage
    #calcul de l'approximation
    for i in range (0,M):
        for j in range (0,N):
            n=-N/2+j
            result[i]+=2**(-abs(n))*(np.exp(1j*n*x[i])).real
    return result

#%% PROJECTION
# Cas T=0 
T=0
M=100
N=16
x=np.arange(0,2*np.pi,2*np.pi/M) #pour tracer la solution projeté
phi_proj=projection(N,M) # projection

# on trace la projection
plt.plot(x,phi_proj)
plt.xlabel("x")
plt.title("Projection pour t=0")
plt.show()
#%% PROJECTION
# Cas T=1
T=1
sol_proj2=CollocFourier(phi_proj,N,nu,dt,T) #calcul en T=1
# on trace la projection
plt.plot(x,sol_proj2)
plt.xlabel("x")
plt.title("Projection pour t=1")
plt.show()
#%% PROJECTION
# Cas T=2
T=2
sol_proj3=CollocFourier(phi_proj,N,nu,dt,T)#calcul en T=2
# on trace la projection
plt.plot(x,sol_proj3)
plt.xlabel("x")
plt.title("Projection pour t=2")
plt.show()

#%% ERREUR

def norme_discrete(x,y):
    # calcul de la norme discrete au carré entre x et y
    M=len(x) # nombre de points
    s=0 #stockage
    # calcul de la norme :
    for i in range (0,M):
        s=s+(x[i]-y[i])**2    
    s=s*2*np.pi/M
    return s

#on calcul les erreurs :
n=np.arange(6,54,4) # pour ces valeurs de N
tt=np.array([0,1,2]) # pour ces valur de T
M=100 
x=np.linspace(0,2*np.pi-h,M)
for ttt in tt : # pour chaque T
    i=0 # initialisation compteur
    stock_inter=np.zeros(len(n)) #stockage
    stock_proj=np.zeros(len(n)) # stockage
    for NN in n : # pour chaque N
        h=np.pi*2/NN # le pas pour N
        Zn=np.linspace(0,2*np.pi-h,NN) # N points
        sol_ex=phi(x,ttt,nu).real #solution exact
        sol_ex_N=phi(Zn,ttt,nu).real #solution exact
        sol_ini=phi0(Zn) # approximation de la solution exact
        temp=CollocFourier(sol_ini,NN,nu,dt,ttt) # approximation en T=ttt
        inter=interp_phi(temp,x,Zn,h) # interpolation 
        phi_proj=projection(NN,NN) # projection
        proj=CollocFourier(phi_proj,NN,nu,dt,ttt) #approx par projection
        erreur_inter=norme_discrete(sol_ex,inter) # erreur par interpolation
        erreur_proj=norme_discrete(sol_ex_N,proj) #erreur par projection
        stock_inter[i]=erreur_inter # on stock
        stock_proj[i]=erreur_proj # on stock
        i+=1 #compteur
    
    #on trace les erreurs
    plt.plot(n,stock_inter, label="interpolation")
    plt.plot(n,stock_proj, label="projection")
    plt.yscale("log")
    plt.ylabel("erreur (en log)")
    plt.xlabel("N")
    plt.title("Erreur discrète selon N pour t="+str(ttt))
    plt.legend()
    plt.show()


#%%

# Question 1 a:
    
def DerConvectionDiffusionTemps(phi_chap,nu):
    # derivee des coefficients phi_chap
    # nu le parametre de l'équation
    n=np.size(phi_chap) #nombre de points
    der=np.zeros(n,dtype=complex) #stockage
    #calcul de derivation :
    for i in range (0,n):
        k=-int((n-1)/2)+i
        der[i]=-(1j*k+nu*k**2)*phi_chap[i]
    return der
    

# Question 1 b:

def PasMethodeGalerkinFourier(dt,phi_chap,nu):
    # execute 1 pas d'une
    # méthode à un pas stable pour la résolution
    # d’un système d’équations différentielles ordinaires d’ordre 1
    # (Runge-Kutta) dans le cas Galerkin Fourier
    # pour passer de t=t_n a t=t_{n+1}=t_n+dt.
    # phi est la solution initiale
    # dt est le pas en temps
    # nu est le parametre de l'équation
    U=phi_chap
    G=DerConvectionDiffusionTemps(phi_chap,nu)
    U=U+(1./3)*dt*G
    G=-(5./9)*G+DerConvectionDiffusionTemps(phi_chap,nu)
    U=U+(15./16)*dt*G
    G=-(153./128)*G+DerConvectionDiffusionTemps(phi_chap,nu)
    return U+(8./15)*dt*G


# Question 1 c :

def CalculApproxGalerkinFourier(x,phi_chap):
    # calcule la valeur de la solution approchée au point x
    # à patir de phi_chap
    n=np.size(phi_chap) # Nombre de points
    sol_approx=0 #initialisation
    # calcul de l'approximation :
    for i in range (0,n):
        k=-int((n-1)/2)+i
        sol_approx=sol_approx+phi_chap[i]*np.exp(1j*k*x).real
    return sol_approx


# Question 2

def FourierGalerkin(N,T,dt,phi_ini,N_out,nu):
    # i intègre l’équation de convection-diffusion dans
    # l’intervalle de temps [0, T] avec un pas dt et condition
    # initiale phi_chap en N points et synthétise la solution en N_out
    # points de [0, T].
    # nu est le parametre de l'equation
    t=0 # initialisayion
    h=2*np.pi/N_out # pas de sortie
    x_i=np.arange(0,2*np.pi+2*np.pi/N-h,2*np.pi/N) # x de départ
    res_int=np.arange(0,2*np.pi+2*np.pi/N_out-h,2*np.pi/N_out) # x de sortie
    

    stock=[] #stockage de valeurs de l'approximation a chaque pas
    #pour t=0  :
    # calcul de l'approximation en N_out points :
    val=CalculApproxGalerkinFourier(x_i,phi_ini)
    # on stock en N_out point l'approximation par interpolation :
    stock.append(interp_phi(val,res_int,x_i,h))
    
    # l'approximation pour chaque t_n :
    while t<=T:
        t=t+dt # = t_n
        phi_ini=PasMethodeGalerkinFourier(dt,phi_ini,nu) #approximation suivante
        val=CalculApproxGalerkinFourier(x_i,phi_ini).real # approximation en x
        stock.append(interp_phi(val,res_int,x_i,h)) # on stock
    
    return stock,res_int

#%% GALERKIN FOURIER
    
# Question 3
    
#calcul phi_chap :
phi_chap=np.zeros(N)
for i in range(N):
    n=-N/2+i
    phi_chap[i]=2**(-abs(n))

#donnees
dt=5*10**(-3)
T=2
N=16
N_out=50
nu=0.2

#on test sur phi_chap :
solution=np.real(FourierGalerkin(N,T,dt,phi_chap,N_out,nu))
#%% GALERKIN FOURIER
#on trace la solution
plt.plot(solution[1],solution[0][0],label="t=0")
plt.plot(solution[1],solution[0][int(len(solution[0])/2)],label="t=1")
plt.plot(solution[1],solution[0][400],label="t=2")
plt.xlabel("x")
plt.title("Approximation Galerkin")
plt.legend()
plt.show()

#%% ERREUR GALERKIN FOURIER


# Question 4


n=np.arange(6,34,4) # les valeurs de N que l'on test
dt=np.array([5*10**(-3),2.5*10**(-3),1.25*10**(-3)]) # les valeurs de dt que l'on test
M=100 # N_out
T=2 # temps final
x=np.linspace(0,2*np.pi-h,M) # x de sortie
for ttt in dt : # pour chaque dt
    i=0 #initialisation du compteur
    
    #stockage :
    stock0=np.zeros(len(n))
    stock1=np.zeros(len(n))
    stock2=np.zeros(len(n))

    for NN in n : # pour chaque N
        h=np.pi*2/NN # le pas pour N points
        Zn=np.linspace(0,2*np.pi-h,NN) # x avec N points
        #solution exact :
        sol_ex0=phi(Zn,0,nu).real # en t=0
        sol_ex1=phi(Zn,1,nu).real # en t=1
        sol_ex2=phi(Zn,2,nu).real # en t=2
        
        sol_ini=phi0(Zn) # approximation de la solution initiale
        #on calcul les approximation en M points:
        approx=FourierGalerkin(NN,T,ttt,sol_ini,M,nu)

        #erreur :
        erreur0=norme_discrete(sol_ex0,approx[0][0]) # t=0
        erreur1=norme_discrete(sol_ex1,approx[0][int(len(approx[0])/2)]) #t=1
        erreur2=norme_discrete(sol_ex2,approx[0][len(approx[0])-1]) # t=2

        #stockage erreur
        stock0[i]=erreur0 # t=0
        stock1[i]=erreur1 # t=1
        stock2[i]=erreur2 # t=2

        i+=1 # compteur

    #on trace les erreurs:
    plt.plot(n,stock0, label="t=0")
    plt.plot(n,stock1, label="t=1")
    plt.plot(n,stock2, label="t=2")
    plt.yscale("log")
    plt.ylabel("erreur (en log)")
    plt.xlabel("N")
    plt.title("Erreur discrète selon N pour dt="+str(ttt))
    plt.legend()
    plt.show()