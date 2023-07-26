#FrozenLake with "Deep (not so deep) Q-learning"

import random
import numpy
import math
from commun import *
from ia import *

epsilon=0.1
gamma=0.9
alpha=0.5

#%%

def sigmoid(x):
    if x>100:
        returnValue=1
    elif x<-100:
        returnValue=-1
    else:
        returnValue=(math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x))
    return returnValue
        
class NN:
    def __init__(self,sizeInput,sizeHiddenLayer,sizeOutput):
        self.sizeInput=sizeInput
        self.sizeHiddenLayer=sizeHiddenLayer
        self.sizeOutput=sizeOutput

        #below are the weights
        self.HiddenLayerEntryWeights=numpy.zeros([sizeHiddenLayer,sizeInput])
        self.LastLayerEntryWeights=numpy.zeros([sizeOutput,sizeHiddenLayer])

        #random initialization
        for i in range(0,sizeHiddenLayer):
            for j in range(0,sizeInput):
                self.HiddenLayerEntryWeights[i,j]=random.uniform(-0.1,0.1)
                
        for i in range(0,sizeOutput):
            for j in range(0,sizeHiddenLayer):
                self.LastLayerEntryWeights[i,j]=random.uniform(-0.1,0.1)

        self.HiddenLayerEntryDeltas=numpy.zeros(sizeHiddenLayer)
        self.LastLayerEntryDeltas=numpy.zeros(sizeOutput)

        self.HiddenLayerOutput=numpy.zeros(sizeHiddenLayer)
        self.LastLayerOutput=numpy.zeros(sizeOutput)

    def output(self,x):
        for i in range(0, self.sizeHiddenLayer):
            self.HiddenLayerOutput[i]=sigmoid(numpy.dot(self.HiddenLayerEntryWeights[i],x))
        for i in range(0, self.sizeOutput):
            self.LastLayerOutput[i]= \
            sigmoid(numpy.dot(self.LastLayerEntryWeights[i],self.HiddenLayerOutput))

    def retropropagation(self,x,y,actionIndex):
        self.output(x)

        #deltas computation
        self.LastLayerEntryDeltas[actionIndex]=2*(self.LastLayerOutput[actionIndex]-y)* \
            (1+self.LastLayerOutput[actionIndex])*(1-self.LastLayerOutput[actionIndex])

        for i in range(0,self.sizeHiddenLayer):
            #here usually you need a sum
            self.HiddenLayerEntryDeltas[i]=self.LastLayerEntryDeltas[actionIndex]* \
            (1+self.HiddenLayerOutput[i])*(1-self.HiddenLayerOutput[i])*self.LastLayerEntryWeights[actionIndex,i]

        #weights update
        for i in range(0,self.sizeHiddenLayer):
            self.LastLayerEntryWeights[actionIndex,i]-=alpha*self.LastLayerEntryDeltas[actionIndex]* \
            self.HiddenLayerOutput[i]

        for i in range(0,self.sizeHiddenLayer):
            for j in range(0,self.sizeInput):
                self.HiddenLayerEntryWeights[i,j]-=alpha*self.HiddenLayerEntryDeltas[i]*x[j]

        
nbCells=6*7
sizeInput=nbCells
sizeHiddenLayer=50
sizeOutput=7
myNN = NN(sizeInput, sizeHiddenLayer, sizeOutput)
cpt_victoire=[0,0,0]
next_cpt_victoire=[0,0,0]


#%% ia vs ia.py
alpha=0.3
nbEpisodes_2=5000

for i in range(0, nbEpisodes_2):
    if i%1000==0:
        myNN_past=myNN
    if i>5000:
        alpha=0.1
    if i>10000:
        alpha=0.05
    if i>20000:
        alpha=0.02
    if i>30000:
        alpha=0.01
    
    if i%100==0:
        print("episode: "+str(i))
        print("  -Score:"+str(cpt_victoire))
        cpt_victoire=[0,0,0]

    state=initialise_liste_positions()

    #--------------------
    tirage_joueur=random.randint(1,2)
    if tirage_joueur==1:
        joueur_actuel="yellow"
    else :
        joueur_actuel="red"
    #--------------------

    
    firstState=state
    endOfEpisode = False
    nbSteps=0
    reward=0

    while not endOfEpisode:
        if joueur_actuel=="yellow":
            if random.uniform(0, 1) < epsilon:
                action = random.randint(1, 7)
                while colonne_pleine(state, action):
                    action = random.randint(1, 7)
            else:
                x=state
                myNN.output(x)
                action = numpy.argmax(myNN.LastLayerOutput)+1
                cpt_action=0
                while colonne_pleine(state, action):
                    cpt_action+=1
                    action=numpy.argsort(-myNN.LastLayerOutput)[cpt_action]+1
                    

            next_state=jouer_bis(state, joueur_actuel, action)  
            endOfEpisode, joueur_suivant, next_cpt_victoire = fin_partie_bis(next_state, joueur_actuel, cpt_victoire) 
            if endOfEpisode==True:
                reward=1
            
        elif joueur_actuel=="red":
            jouer_ordi_ia(state, "red", 15)
            endOfEpisode, joueur_suivant, next_cpt_victoire = fin_partie_bis(state, joueur_actuel, cpt_victoire)
            if endOfEpisode==True:
                reward=-1

        
        x=state
            
        myNN.output(x)
        next_max = numpy.max(myNN.LastLayerOutput)

        target = reward+gamma*next_max
   
        x=state
        myNN.retropropagation(x,target,action-1)
  
        state = next_state
        joueur_actuel=joueur_suivant    
    
        cpt_victoire=next_cpt_victoire
        nbSteps=nbSteps+1
        
print("end of learning period")

#%% ia vs rand


alpha=0.5
nbEpisodes=5000

for i in range(0, nbEpisodes):
    if i>5000:
        alpha=0.1
    if i>10000:
        alpha=0.05
    if i>20000:
        alpha=0.02
    if i>30000:
        alpha=0.01
    
    if i%100==0:
        print("episode: "+str(i))
        print("  -Score:"+str(cpt_victoire))
        cpt_victoire=[0,0,0]

    state=initialise_liste_positions()
    
    #--------------------
    tirage_joueur=random.randint(1,2)
    if tirage_joueur==1:
        joueur_actuel="yellow"
    else :
        joueur_actuel="red"
    #--------------------
    
    firstState=state
    endOfEpisode = False
    nbSteps=0

    while not endOfEpisode:
        if joueur_actuel=="yellow":
            if random.uniform(0, 1) < epsilon:
                action = random.randint(1, 7)
                while colonne_pleine(state, action):
                    action = random.randint(1, 7)
            else:
                x=state
                myNN.output(x)
                action = numpy.argmax(myNN.LastLayerOutput)+1
                cpt_action=0
                while colonne_pleine(state, action):
                    cpt_action+=1
                    action=numpy.argsort(-myNN.LastLayerOutput)[cpt_action]+1
                    

            next_state=jouer_bis(state, joueur_actuel, action)       
            endOfEpisode, joueur_suivant, next_cpt_victoire = fin_partie_bis(next_state, joueur_actuel, cpt_victoire) 
            
        elif joueur_actuel=="red":
            action = random.randint(1, 7)
            while colonne_pleine(state, action):
                action = random.randint(1, 7)
            
            next_state=jouer_bis(state, joueur_actuel, action)
            endOfEpisode, joueur_suivant, next_cpt_victoire = fin_partie_bis(state, joueur_actuel, cpt_victoire)
            

        if (next_cpt_victoire[0]-cpt_victoire[0])==1:
            reward=1
        elif (next_cpt_victoire[1]-cpt_victoire[1])==1:
            reward=-1
        else :
            reward=0

        x=state
            
        myNN.output(x)
        next_max = numpy.max(myNN.LastLayerOutput)

        target = reward+gamma*next_max
   
        x=state
        myNN.retropropagation(x,target,action-1)
  
        state = next_state
        joueur_actuel=joueur_suivant    
    
        cpt_victoire=next_cpt_victoire
        nbSteps=nbSteps+1
print("end of learning period")

#%% ia vs ia_past1000
myNN_past=myNN

alpha=0.5
nbEpisodes_2=5000

for i in range(0, nbEpisodes_2):
    if i%1000==0:
        myNN_past=myNN
    if i>5000:
        alpha=0.1
    if i>10000:
        alpha=0.05
    if i>20000:
        alpha=0.02
    if i>30000:
        alpha=0.01
    
    if i%100==0:
        print("episode: "+str(i))
        print("  -Score:"+str(cpt_victoire))
        cpt_victoire=[0,0,0]

    state=initialise_liste_positions()
    
    #--------------------
    tirage_joueur=random.randint(1,2)
    if tirage_joueur==1:
        joueur_actuel="yellow"
    else :
        joueur_actuel="red"
    #--------------------

    
    firstState=state
    endOfEpisode = False
    nbSteps=0

    while not endOfEpisode:
        if joueur_actuel=="yellow":
            if random.uniform(0, 1) < epsilon:
                action = random.randint(1, 7)
                while colonne_pleine(state, action):
                    action = random.randint(1, 7)
            else:
                x=state
                myNN.output(x)
                action = numpy.argmax(myNN.LastLayerOutput)+1
                cpt_action=0
                while colonne_pleine(state, action):
                    cpt_action+=1
                    action=numpy.argsort(-myNN.LastLayerOutput)[cpt_action]+1
                    

            next_state=jouer_bis(state, joueur_actuel, action)       
            endOfEpisode, joueur_suivant, next_cpt_victoire = fin_partie_bis(next_state, joueur_actuel, cpt_victoire) 
            
        elif joueur_actuel=="red":
            x=state
            myNN_past.output(x)
            action = numpy.argmax(myNN_past.LastLayerOutput)+1
            cpt_action=0
            while colonne_pleine(state, action):
                cpt_action+=1
                action=numpy.argsort(-myNN_past.LastLayerOutput)[cpt_action]+1
                    
            
            next_state=jouer_bis(state, joueur_actuel, action)
            endOfEpisode, joueur_suivant, next_cpt_victoire = fin_partie_bis(state, joueur_actuel, cpt_victoire)
        
        
            
        if (next_cpt_victoire[0]-cpt_victoire[0])==1:
            reward=1
        elif (next_cpt_victoire[1]-cpt_victoire[1])==1:
            reward=-1
        else :
            reward=0
        x=state
            
        myNN.output(x)
        next_max = numpy.max(myNN.LastLayerOutput)

        target = reward+gamma*next_max
   
        x=state
        myNN.retropropagation(x,target,action-1)
  
        state = next_state
        joueur_actuel=joueur_suivant    
    
        cpt_victoire=next_cpt_victoire
        nbSteps=nbSteps+1
        
print("end of learning period")


#%% test ia vs random
nbEpisodes=100
cpt_vic=[0,0,0]
next_cpt_vic=[0,0,0]
for i in range(0, nbEpisodes):
    if i%50==0:
        print("episode: "+str(i))
        successesInARow=0

    state=initialise_liste_positions()
    
    #--------------------
    tirage_joueur=random.randint(1,2)
    if tirage_joueur==1:
        joueur_actuel="yellow"
    else :
        joueur_actuel="red"
    #--------------------
    
    firstState=state
    endOfEpisode = False
    nbSteps=0

    while not endOfEpisode:
        if joueur_actuel=="yellow":
            x=state
            myNN.output(x)
            action = numpy.argmax(myNN.LastLayerOutput)+1
            #action = random.randint(1, 7)
            cpt_action=0
            while colonne_pleine(state, action):
                cpt_action+=1
                action=numpy.argsort(-myNN.LastLayerOutput)[cpt_action]+1

            next_state=jouer_bis(state, joueur_actuel, action)

            endOfEpisode, joueur_suivant, next_cpt_vic = fin_partie_bis(next_state, joueur_actuel, cpt_vic)
            
            x=state
            
            state = next_state
            joueur_actuel=joueur_suivant

        elif joueur_actuel=="red":
            action = random.randint(1, 7)
            while colonne_pleine(state, action):
                action = random.randint(1, 7)
            
            next_state=jouer_bis(state, joueur_actuel, action)
            endOfEpisode, joueur_suivant, next_cpt_vic = fin_partie_bis(next_state, joueur_actuel, cpt_vic)

            state = next_state
            joueur_actuel=joueur_suivant
            
        if (next_cpt_vic[0]-cpt_vic[0])==1:
            reward=1
        elif (next_cpt_vic[1]-cpt_vic[1])==1:
            reward=-1
        else :
            reward=0

        cpt_vic=next_cpt_vic
        
        nbSteps=nbSteps+1

print(cpt_vic)
