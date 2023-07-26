#Projet 1 Méthodes Numériques pour l'Ingénieur

#Simao Valentin, Renuy Florian, Desrumaux Jonathan

# Partie 5

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import *
from mpl_toolkits.mplot3d import Axes3D


#%%
N=48
nbsteps=300
h=2*math.pi/N           #Définit le pas
xm=np.arange(h,2*math.pi+h,h)                  #Pour définir les x_i

T=8
delta_t=T/nbsteps       #Définir le pas du temps
t=0
u=[t]*N
time=[u]

while t < T:
    t=t+delta_t
    u=[t]*N
    time.append(u)
 #%%   
    




def SN(x):
    if x==0:            #On définie la fonction sinc périodique
        return 1
    else:
        return math.sin(math.pi*x/h)/(2*math.pi*math.tan(x/2)/h)

def p(liste):
    p=[]
    for k in range(len(xm)):     #On calcule la transformé de Fourier discrète
        valeur=0                #d'une liste, terme par terme, qu'on stock dans
        for m in range(N):      #la liste p
            valeur=valeur+liste[m]*SN(xm[k]-xm[m])
        p.append(valeur)        #On a en sortie le vecteur P
    return p


def deriv_p(liste,N):         #On calcule DN*P pour avoir P'
    DN=np.zeros((N,N))
    for i in range(N):
        for j in range(N):      #On créé la matrice DN puis on la remplie
            if i==j:
                DN[i][j]=0
                    
            else:
                DN[i][j]=(1/2)*(1/math.tan((i-j)*h/2))*(-1)**(i-j)
    deriv_p=DN.dot(liste)
    return deriv_p


def DerSpecPer(liste,N):
    return deriv_p(p(liste),N)       #Fonction qui calcule la dérivation
                             

def u(liste,n):
    u=np.exp(-100*(liste-1+0.2*n)**2)   #Fonction qui calcul u(xm)
    return u


def c(liste):                       #Fonction qui calcul c(x)
    return 0.2+(np.sin(liste-1))**2


#%%

stock=[]        #pour le graphe 3d


for n in range(nbsteps): #à l'étape n (on commence à n=0) 
    if n==0:
        v_hold=u(xm,-delta_t)
        v_actual=u(xm,0)
        stock.append(v_hold)
        stock.append(v_actual)

    stock.append(stock[-2]-2*delta_t*c(xm)*DerSpecPer(stock[-1],N))        #pour le graphe 3d


#%%
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')      #Graphe 3D
    
for i in range(len(time)):          #Trace toutes les courbes en 3D
    plt.plot(xm,time[i],stock[i],color="black",linewidth=0.25)

plt.title("Graphe d'une onde de période T=8")
ax.set_zlim3d(0,20)     
ax.view_init(30, -80)       #Orientation du graphe 3D
plt.show()


