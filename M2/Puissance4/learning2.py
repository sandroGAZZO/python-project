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

epsilon=0.1
gamma=0.99
alpha=0.00001
epsilonDecay=0.99

#sigmoid going from -1 to 1
def sigmoid(x):
    if x>100:
        returnValue=1
    elif x<-100:
        returnValue=-1
    else:
        returnValue=(math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x))
    return returnValue

def sigmoidDerivative(x):
    if x>100:
        returnValue=0
    elif x<-100:
        returnValue=0
    else:
        returnValue=(1+sigmoid(x))*(1-sigmoid(x))
    return returnValue

def softPlus(x):
    if x>100:
        returnValue=x
    elif x<-100:
        returnValue=0
    else:
        returnValue=math.log(1+math.exp(x))
    return returnValue

def softPlusDerivative(x):
    if x>100:
        returnValue=1
    elif x<-100:
        returnValue=0
    else:
        returnValue=1/(1+math.exp(-x))
    return returnValue

            
class NN:
    def __init__(self,sizeInput,sizeHiddenLayer1,sizeHiddenLayer2,sizeOutput,lastLayerActivationType):
        self.sizeInput=sizeInput
        self.sizeHiddenLayer1=sizeHiddenLayer1
        self.sizeHiddenLayer2=sizeHiddenLayer2
        self.sizeOutput=sizeOutput
        self.LastLayerActivationType=lastLayerActivationType

        #below are the weights
        self.HiddenLayer1EntryWeights=numpy.zeros([sizeHiddenLayer1,sizeInput])
        self.HiddenLayer2EntryWeights=numpy.zeros([sizeHiddenLayer2,sizeHiddenLayer1])
        self.LastLayerEntryWeights=numpy.zeros([sizeOutput,sizeHiddenLayer2])

        a=0.01
        
        #random initialization
        for i in range(0,sizeHiddenLayer1):
            for j in range(0,sizeInput):
                self.HiddenLayer1EntryWeights[i,j]=random.uniform(-a,a)

        for i in range(0,sizeHiddenLayer2):
            for j in range(0,sizeHiddenLayer1):
                self.HiddenLayer2EntryWeights[i,j]=random.uniform(-a,a)
                
        for i in range(0,sizeOutput):
            for j in range(0,sizeHiddenLayer2):
                self.LastLayerEntryWeights[i,j]=random.uniform(-a,a)

        self.HiddenLayer1EntryDeltas=numpy.zeros(sizeHiddenLayer1)
        self.HiddenLayer2EntryDeltas=numpy.zeros(sizeHiddenLayer2)
        self.LastLayerEntryDeltas=numpy.zeros(sizeOutput)

        self.HiddenLayer1Output=numpy.zeros(sizeHiddenLayer1)
        self.HiddenLayer2Output=numpy.zeros(sizeHiddenLayer2)
        self.LastLayerOutput=numpy.zeros(sizeOutput)

    def output(self,x):
        for i in range(0, self.sizeHiddenLayer1):
            self.HiddenLayer1Output[i]=sigmoid(numpy.dot(self.HiddenLayer1EntryWeights[i],x))
        for i in range(0, self.sizeHiddenLayer2):
           self.HiddenLayer2Output[i]=sigmoid(numpy.dot(self.HiddenLayer2EntryWeights[i],self.HiddenLayer1Output))
        for i in range(0, self.sizeOutput):
            self.LastLayerOutput[i]=numpy.dot(self.LastLayerEntryWeights[i],self.HiddenLayer2Output)
            if self.LastLayerActivationType[i]=="sigmoid":
                self.LastLayerOutput[i]=sigmoid(self.LastLayerOutput[i])
            elif self.LastLayerActivationType[i]=="positiveSigmoid":
                self.LastLayerOutput[i]=(sigmoid(self.LastLayerOutput[i])+1)/2
            elif self.LastLayerActivationType[i]=="softPlus":
            	self.LastLayerOutput[i]=softPlus(self.LastLayerOutput[i])
            #no activation

    #écrite pour minimiser la perte, alpha négatif permet de l'augmenter
    def retropropagation(self,x,y,actionIndex,alpha):
        self.output(x)

        #deltas computation
        if self.LastLayerActivationType[actionIndex]=="sigmoid":
            self.LastLayerEntryDeltas[actionIndex]=2*(self.LastLayerOutput[actionIndex]-y)* \
            sigmoidDerivative(self.LastLayerOutput[actionIndex])
        elif self.LastLayerActivationType[actionIndex]=="positiveSigmoid":
            self.LastLayerEntryDeltas[actionIndex]=2*(self.LastLayerOutput[actionIndex]-y)*0.5* \
            sigmoidDerivative(numpy.dot(self.LastLayerEntryWeights[actionIndex],self.HiddenLayer2Output))
        elif self.LastLayerActivationType[actionIndex]=="softPlus":
            self.LastLayerEntryDeltas[actionIndex]=2*(self.LastLayerOutput[actionIndex]-y)* \
            softPlusDerivative(numpy.dot(self.LastLayerEntryWeights[actionIndex],self.HiddenLayer2Output))
        else:
            #no activation
            self.LastLayerEntryDeltas[actionIndex]=2*(self.LastLayerOutput[actionIndex]-y)
                
        for i in range(0,self.sizeHiddenLayer2):
            #here usually you need a sum
            self.HiddenLayer2EntryDeltas[i]=self.LastLayerEntryDeltas[actionIndex]* \
                    (1+self.HiddenLayer2Output[i])*(1-self.HiddenLayer2Output[i])*self.LastLayerEntryWeights[actionIndex,i]

        for i in range(0,self.sizeHiddenLayer1):
            #here usually you need a sum
            self.HiddenLayer1EntryDeltas[i]=0
            for j in range(0,self.sizeHiddenLayer2):
                self.HiddenLayer1EntryDeltas[i]+=self.HiddenLayer2EntryDeltas[j]* \
                    (1+self.HiddenLayer1Output[i])*(1-self.HiddenLayer1Output[i])*self.HiddenLayer2EntryWeights[j,i]
                    
        #weights update
        for i in range(0,self.sizeHiddenLayer2):
            self.LastLayerEntryWeights[actionIndex,i]-=alpha*self.LastLayerEntryDeltas[actionIndex]* \
                    self.HiddenLayer2Output[i]

        for i in range(0,self.sizeHiddenLayer1):
            for j in range(0,self.sizeHiddenLayer2):
                self.HiddenLayer2EntryWeights[j,i]-=alpha*self.HiddenLayer2EntryDeltas[j]* \
                    self.HiddenLayer1Output[i]
            
        for i in range(0,self.sizeHiddenLayer1):
            for j in range(0,self.sizeInput):
                self.HiddenLayer1EntryWeights[i,j]-=alpha*self.HiddenLayer1EntryDeltas[i]*x[j]

nbCells=6*7
sizeInput=nbCells
sizeHiddenLayer1=64
sizeHiddenLayer2=64
sizeOutput=7
myNN = NN(sizeInput, sizeHiddenLayer1, sizeHiddenLayer2, sizeOutput,["sigmoid","sigmoid","sigmoid","sigmoid","sigmoid","sigmoid","sigmoid"])


cpt_victoire=[0,0,0]
next_cpt_victoire=[0,0,0]


#%% ia vs ia.py
alpha=0.5
nbEpisodes_2=50000

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
            jouer_ordi_ia(state, "red", 15)
            endOfEpisode, joueur_suivant, next_cpt_victoire = fin_partie_bis(state, joueur_actuel, cpt_victoire)
            
        if (next_cpt_victoire[0]-cpt_victoire[0])==1:
            reward=1
        elif (next_cpt_victoire[1]-cpt_victoire[1])==1:
            reward=-1
        else:
            reward=0

        x=state
            
        myNN.output(x)
        next_max = numpy.max(myNN.LastLayerOutput)

        target = reward+gamma*next_max
   
        x=state
        myNN.retropropagation(x,target,action-1,alpha)
  
        state = next_state
        joueur_actuel=joueur_suivant    
    
        cpt_victoire=next_cpt_victoire
        nbSteps=nbSteps+1
        
print("end of learning period")

#%% ia vs rand


alpha=0.5
nbEpisodes=500

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
        else:
            reward=0

        x=state
            
        myNN.output(x)
        next_max = numpy.max(myNN.LastLayerOutput)

        target = reward+gamma*next_max
   
        x=state
        myNN.retropropagation(x,target,action-1,alpha)
  
        state = next_state
        joueur_actuel=joueur_suivant    
    
        cpt_victoire=next_cpt_victoire
        nbSteps=nbSteps+1
print("end of learning period")

#%% ia vs ia_past1000
myNN_past=myNN
alpha=0.2
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
        else:
            reward=0
            
        x=state
            
        myNN.output(x)
        next_max = numpy.max(myNN.LastLayerOutput)

        target = reward+gamma*next_max
   
        x=state
        myNN.retropropagation(x,target,action-1,alpha)
  
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
