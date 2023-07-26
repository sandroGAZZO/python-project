import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import *
from mpl_toolkits.mplot3d import Axes3D 

#%% Question 5

t=np.linspace(0,T,len(reso[1,:])) #on définit le temps de 0 a 8 avec un pas de dt

XX, TT= np.meshgrid(x,t) # on définit les points de la grille sur X,Y

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')      #Graphe 3D

for i in range(len(TT)):          #Trace toutes les courbes en 3D
    plt.plot(x,TT[i],reso[:,i],color="black",linewidth=0.25)

plt.title("Propagation de l'onde")
ax.set_zlim3d(0,5)     # limite de l'axe z
ax.view_init(30, -80)#Orientation du graphe 3D
plt.ylabel("temps")
plt.xlabel("x")
plt.show()


























