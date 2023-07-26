import math
import numpy
import random 
from scipy.optimize import linprog

#donne le numéro de l'état
def calculEtat(joueur1x,joueur1y,joueur2x,joueur2y,ballon):
    positionJoueur1=joueur1x+5*joueur1y
    positionJoueur2=joueur2x+5*joueur2y
    if ballon=="joueur1":
        etatCourant=positionJoueur1*20+positionJoueur2
    else:
        etatCourant=400+positionJoueur1*20+positionJoueur2

    #if etatCourant>=800:
    #    print(str(positionJoueur1)+" ; "+str(positionJoueur2))
    #    print(str(joueur1x)+" ; "+str(joueur1y)+" ; "+str(joueur2x)+" ; "+str(joueur2y)+" ; "+ballon)
    return etatCourant

Q=numpy.zeros([800,5,5])
pi=numpy.zeros([800,5])
for i in range(0,800):
    for j in range(0,5):
        pi[i,j]=1.0/5

V=numpy.zeros(800)
nbIterations=100000


#actions codées ainsi : 0 haut, 1 droite, 2 bas, 3 gauche et 4 immobile
haut=0
droite=1
bas=2
gauche=3
immobile=4

epsilon=0.2
alpha=0.01
gamma=1.0
decay=0.9999954

#c'est joueur1 qui apprend
sommeScores=0
for iteration in range(0,nbIterations):
    print("iteration : "+str(iteration)) 
    #positionInitiale
    joueur1x=3
    joueur1y=2
    joueur2x=1
    joueur2y=1

    if random.randint(0,1)==0:
        ballon="joueur1"
    else:
        ballon="joueur2"
    partieFinie=False

    etatCourant=calculEtat(joueur1x,joueur1y,joueur2x,joueur2y,ballon)
    while partieFinie==False:
        score=0
        if random.uniform(0,1)<epsilon:
            actionJoueur1=random.randint(0,4)
        else:
            u=random.uniform(0,1)
            if u<sum(pi[etatCourant][0:1]):
                actionJoueur1=haut
            elif u<sum(pi[etatCourant][0:2]):
                actionJoueur1=droite
            elif u<sum(pi[etatCourant][0:3]):
                actionJoueur1=bas
            elif u<sum(pi[etatCourant][0:4]):
                actionJoueur1=gauche
            else:
                actionJoueur1=immobile
        actionJoueur2=random.randint(0,4)
        
        #on contrôle les actions illicites
        if joueur1x==0 and actionJoueur1==gauche:
            actionJoueur1=immobile
        if joueur1x==4 and actionJoueur1==droite:
            actionJoueur1=immobile
        if joueur1y==0 and actionJoueur1==bas:
            actionJoueur1=immobile
        if joueur1y==3 and actionJoueur1==haut:
            actionJoueur1=immobile
        if joueur2x==0 and actionJoueur2==gauche:
            actionJoueur2=immobile
        if joueur2x==4 and actionJoueur2==droite:
            actionJoueur2=immobile
        if joueur2y==0 and actionJoueur2==bas:
            actionJoueur2=immobile
        if joueur2y==3 and actionJoueur2==haut:
            actionJoueur2=immobile

        nballon=ballon
        if random.randint(0,1)==0:
            #action du joueur 1 d'abord
            if actionJoueur1==haut:
                njoueur1x=joueur1x
                njoueur1y=joueur1y+1
            elif actionJoueur1==bas:
                njoueur1x=joueur1x
                njoueur1y=joueur1y-1
            elif actionJoueur1==gauche:
                njoueur1x=joueur1x-1
                njoueur1y=joueur1y
            elif actionJoueur1==droite:
                njoueur1x=joueur1x+1
                njoueur1y=joueur1y
            else:
                njoueur1x=joueur1x
                njoueur1y=joueur1y
            
            if njoueur1x==joueur2x and njoueur1y==joueur2y:
                njoueur1x=joueur1x
                njoueur1y=joueur1y
                if ballon=="joueur1":
                    nballon="joueur2"
            elif ballon=="joueur1" and njoueur1x==0 and (njoueur1y in range(1,3)):
                partieFinie=True
                score=1
            
            #puis action du joueur2
            if actionJoueur2==haut:
                njoueur2x=joueur2x
                njoueur2y=joueur2y+1
            elif actionJoueur2==bas:
                njoueur2x=joueur2x
                njoueur2y=joueur2y-1
            elif actionJoueur2==gauche:
                njoueur2x=joueur2x-1
                njoueur2y=joueur2y
            elif actionJoueur2==droite:
                njoueur2x=joueur2x+1
                njoueur2y=joueur2y
            else:
                njoueur2x=joueur2x
                njoueur2y=joueur2y
            
            if njoueur2x==njoueur1x and njoueur2y==njoueur1y:
                njoueur2x=joueur2x
                njoueur2y=joueur2y
                if nballon=="joueur2":
                    nballon="joueur1"
            elif nballon=="joueur2" and njoueur2x==4 and (njoueur2y in range(1,3)):
                partieFinie=True
                score=-1
        else:
            #action du joueur 2 d'abord
            if actionJoueur2==haut:
                njoueur2x=joueur2x
                njoueur2y=joueur2y+1
            elif actionJoueur2==bas:
                njoueur2x=joueur2x
                njoueur2y=joueur2y-1
            elif actionJoueur2==gauche:
                njoueur2x=joueur2x-1
                njoueur2y=joueur2y
            elif actionJoueur2==droite:
                njoueur2x=joueur2x+1
                njoueur2y=joueur2y
            else:
                njoueur2x=joueur2x
                njoueur2y=joueur2y
            
            if njoueur2x==joueur1x and njoueur2y==joueur1y:
                njoueur2x=joueur2x
                njoueur2y=joueur2y
                if nballon=="joueur2":
                    nballon="joueur1"
            elif ballon=="joueur2" and njoueur2x==4 and (njoueur2y in range(1,3)):
                partieFinie=True
                score=-1
            
            #puis action du joueur 1
            if actionJoueur1==haut:
                njoueur1x=joueur1x
                njoueur1y=joueur1y+1
            elif actionJoueur1==bas:
                njoueur1x=joueur1x
                njoueur1y=joueur1y-1
            elif actionJoueur1==gauche:
                njoueur1x=joueur1x-1
                njoueur1y=joueur1y
            elif actionJoueur1==droite:
                njoueur1x=joueur1x+1
                njoueur1y=joueur1y
            else:
                njoueur1x=joueur1x
                njoueur1y=joueur1y
                
            if njoueur1x==njoueur2x and njoueur1y==njoueur2y:
                njoueur1x=joueur1x
                njoueur1y=joueur1y
                if nballon=="joueur1":
                    nballon="joueur2"
            elif nballon=="joueur1" and njoueur1x==0 and (njoueur1y in range(1,3)):
                partieFinie=True
                score=1

        prochainEtat=calculEtat(njoueur1x,njoueur1y,njoueur2x,njoueur2y,nballon)

        Q[etatCourant,actionJoueur1,actionJoueur2]=(1-alpha)*Q[etatCourant,actionJoueur1,actionJoueur2]+alpha*(score+gamma*V[prochainEtat])

        #problème d'optimisation
        obj = [-1,0,0,0,0,0]
        lhs_ineq=[]
        onOptimise=True
        for i in range(0,5):
            coeffs=[1]
            sommeValeursAbsolues=0
            for j in range(0,5):
                coeffs.append(-Q[etatCourant,j,i])
                sommeValeursAbsolues+=abs(Q[etatCourant,j,i])
            if sommeValeursAbsolues==0:
                onOptimise=False
            lhs_ineq.append(coeffs)
        #print(lhs_ineq)
        rhs_ineq = [0,0,0,0,0]
        lhs_eq=[[0,1,1,1,1,1]]
        rhs_eq=[1]
        bnd = [(-math.inf,math.inf), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)] 

        if onOptimise==True:
            opt = linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq,A_eq=lhs_eq, b_eq=rhs_eq, bounds=bnd,method="revised simplex")
            for i in range(0,5):
                pi[etatCourant,i]=opt.x[i+1]
#            print(opt.fun)
#            print(opt.x)
        else:
            for i in range(0,5):
                pi[etatCourant,i]=1.0/5
        
        #mise-à-jour de V
        V[etatCourant]=math.inf
        for i in range(0,5):
            temp=0
            for j in range(0,5):
                temp+=pi[etatCourant,j]*Q[etatCourant,j,i]
            if temp<V[etatCourant]:
                V[etatCourant]=temp
        
        joueur1x=njoueur1x;joueur1y=njoueur1y
        joueur2x=njoueur2x;joueur2y=njoueur2y
        ballon=nballon
        etatCourant=prochainEtat
#        alpha=alpha*decay
    sommeScores+=score
    print("scoreMoyen : "+str(sommeScores/(1+iteration)))


    
