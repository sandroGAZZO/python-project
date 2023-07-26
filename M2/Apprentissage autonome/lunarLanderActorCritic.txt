import gym
import numpy
import random
import math

#Remarque : il faut installer "pip install box2d box2d-kengz"

env = gym.make('LunarLanderContinuous-v2')
print(env.observation_space)
print(env.action_space)

#2D en action, 8D pour les états

#sigmoid going from -1 to 1
def tanh(x):
    if x>100:
        returnValue=1
    elif x<-100:
        returnValue=-1
    else:
        returnValue=(math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x))
    return returnValue

def tanhDerivative(x):
    if x>100:
        returnValue=0
    elif x<-100:
        returnValue=0
    else:
        returnValue=(1+tanh(x))*(1-tanh(x))
    return returnValue

#sigmoid going from 0 to 1
def sigmoid(x):
    if x>100:
        returnValue=1
    elif x<-100:
        returnValue=0
    else:
        returnValue=math.exp(x)/(1+math.exp(x))
    return returnValue

def sigmoidDerivative(x):
    if x>100:
        returnValue=0
    elif x<-100:
        returnValue=0
    else:
        temp=1+math.exp(x)
        returnValue=math.exp(x)/(temp*temp)
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

        a=0.05
        
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
            self.HiddenLayer1Output[i]=tanh(numpy.dot(self.HiddenLayer1EntryWeights[i],x))
        for i in range(0, self.sizeHiddenLayer2):
           self.HiddenLayer2Output[i]=tanh(numpy.dot(self.HiddenLayer2EntryWeights[i],self.HiddenLayer1Output))
        for i in range(0, self.sizeOutput):
            self.LastLayerOutput[i]=numpy.dot(self.LastLayerEntryWeights[i],self.HiddenLayer2Output)
            if self.LastLayerActivationType[i]=="tanh":
                self.LastLayerOutput[i]=tanh(self.LastLayerOutput[i])
            elif self.LastLayerActivationType[i]=="sigmoid":
                self.LastLayerOutput[i]=sigmoid(self.LastLayerOutput[i])
            elif self.LastLayerActivationType[i]=="softPlus":
            	self.LastLayerOutput[i]=softPlus(self.LastLayerOutput[i])
            #no activation

    #écrite pour minimiser la perte, alpha négatif permet de l'augmenter
    def retropropagation(self,x,differences,alpha):
        self.output(x)

        #deltas computation
        for i in range(0,self.sizeOutput):
            if self.LastLayerActivationType[i]=="tanh":
                self.LastLayerEntryDeltas[i]=differences[i]* \
                tanhDerivative(numpy.dot(self.LastLayerEntryWeights[i],self.HiddenLayer2Output))
            elif self.LastLayerActivationType[i]=="sigmoid":
                self.LastLayerEntryDeltas[i]=differences[i]* \
                sigmoidDerivative(numpy.dot(self.LastLayerEntryWeights[i],self.HiddenLayer2Output))
            elif self.LastLayerActivationType[i]=="softPlus":
                self.LastLayerEntryDeltas[i]=differences[i]* \
                softPlusDerivative(numpy.dot(self.LastLayerEntryWeights[i],self.HiddenLayer2Output))
            else:
            	#no activation
                self.LastLayerEntryDeltas[i]=differences[i]
                
        for i in range(0,self.sizeHiddenLayer2):
            #here usually you need a sum
            self.HiddenLayer2EntryDeltas[i]=0
            for j in range(0,self.sizeOutput):
                self.HiddenLayer2EntryDeltas[i]+=self.LastLayerEntryDeltas[j]* \
                    (1+self.HiddenLayer2Output[i])*(1-self.HiddenLayer2Output[i])*self.LastLayerEntryWeights[j,i]

        for i in range(0,self.sizeHiddenLayer1):
            #here usually you need a sum
            self.HiddenLayer1EntryDeltas[i]=0
            for j in range(0,self.sizeHiddenLayer2):
                self.HiddenLayer1EntryDeltas[i]+=self.HiddenLayer2EntryDeltas[j]* \
                    (1+self.HiddenLayer1Output[i])*(1-self.HiddenLayer1Output[i])*self.HiddenLayer2EntryWeights[j,i]
                    
        #weights update
        for i in range(0,self.sizeHiddenLayer2):
            for j in range(0,self.sizeOutput):
                self.LastLayerEntryWeights[j,i]+=alpha*self.LastLayerEntryDeltas[j]* \
                self.HiddenLayer2Output[i]
                    
        for i in range(0,self.sizeHiddenLayer1):
            for j in range(0,self.sizeHiddenLayer2):
                self.HiddenLayer2EntryWeights[j,i]+=alpha*self.HiddenLayer2EntryDeltas[j]* \
                self.HiddenLayer1Output[i]
                    
        for i in range(0,self.sizeHiddenLayer1):
            for j in range(0,self.sizeInput):
                self.HiddenLayer1EntryWeights[i,j]+=alpha*self.HiddenLayer1EntryDeltas[i]*x[j]


sizeInput=8
#150 50 à gauche
sizeHiddenLayer1=64
sizeHiddenLayer2=64
sizeOutputActionNN=2
sizeOutputValueNN=1
gamma=0.99

actionNN=NN(sizeInput,sizeHiddenLayer1,sizeHiddenLayer2,sizeOutputActionNN,["tanh","tanh"])
valueNN=NN(sizeInput,sizeHiddenLayer1,sizeHiddenLayer2,sizeOutputValueNN,["noActivation"])

nbEpisodes=5000
action=[0,0]
target=[0,0]
moyenne=0

tailleFenetreGlissante=50
derniersScores=[-200]*tailleFenetreGlissante
onRaffine=False
fin=False
nbSteps=0
sigma1=1.0
sigma2=1.0

for i in range(0, nbEpisodes):
    if i%100==1:
        print("episode: "+str(i))
        print(moyenne/i)
        print("########################################################")

    state=env.reset()
    
    firstState=state
    endOfEpisode = False

    #d'abord on simule
    scoreEpisode=0
    I=1
    while not endOfEpisode:
        #calcul de la moyenne glissante
        moyenneGlissante=0
        for j in range(tailleFenetreGlissante-10,tailleFenetreGlissante):
            moyenneGlissante+=derniersScores[j]
        moyenneGlissante/=10

        if moyenneGlissante>-50:
            onRaffine=True
            env.render()

        moyenneGlissante=0
        for j in range(0,tailleFenetreGlissante):
            moyenneGlissante+=derniersScores[j]
        moyenneGlissante/=tailleFenetreGlissante

        if moyenneGlissante>50:
            fin=True
            break
        
        actionNN.output(state)
        if sigma1<0.001:
            sigma1=0.001
        if sigma2<0.001:
            sigma2=0.001  

        action[0]=random.gauss(actionNN.LastLayerOutput[0],sigma1)
        action[1]=random.gauss(actionNN.LastLayerOutput[1],sigma2)

        next_state, reward, endOfEpisode, info = env.step(action) 
        differences=[0,0]

        valueNN.output(state)
        currentStateValue=valueNN.LastLayerOutput[0]

        valueNN.output(next_state)
        nextStateValue=valueNN.LastLayerOutput[0]

        pas=0.0005
        if onRaffine==True:
            #env.render()
            pas=0.0001

        differenceAppreciation=reward+gamma*nextStateValue-currentStateValue

#        if differenceAppreciation>10:
#            differenceAppreciation=10
#        elif differenceAppreciation<-10:
#            differenceAppreciation=-10
        alpha=pas*I*differenceAppreciation

        differences[0]=(action[0]-actionNN.LastLayerOutput[0])/pow(sigma1,2)
        differences[1]=(action[1]-actionNN.LastLayerOutput[1])/pow(sigma2,2)

        sigma1+=(-1/sigma1+(action[0]-actionNN.LastLayerOutput[0])* \
        (action[0]-actionNN.LastLayerOutput[0])/pow(sigma1,3))*alpha
        sigma2+=(-1/sigma2+(action[1]-actionNN.LastLayerOutput[1])* \
        (action[1]-actionNN.LastLayerOutput[1])/pow(sigma2,3))*alpha

        if sigma1>1.0:
            sigma1=1.0
        if sigma2>1.0:
            sigma2=1.0
        
        actionNN.retropropagation(state,differences,alpha)

        alphaValue=pas
        differences[0]=reward+gamma*nextStateValue-currentStateValue

 #       if differences[0]>10.0:
 #           differences[0]=10.0
 #       elif differences[0]<-10.0:
 #           differences[0]=-10.0
        #print(differences[0])
        valueNN.retropropagation(state,differences,alphaValue)        
        
        I=gamma*I
        state = next_state
        scoreEpisode+=reward

        nbSteps+=1

    if i%100==1:
            print(str(sigma1)+" "+str(sigma2))
        

    if fin==True:
        break
        
    if i<tailleFenetreGlissante:
        derniersScores[i]=scoreEpisode
    else:
        derniersScores.pop(0)
        derniersScores.append(scoreEpisode)

    moyenne+=scoreEpisode
    print(scoreEpisode)


   


    
print("end of learning period")

env.close()



#evaluation
recompenseMoyenne=0
nbEpisodes=100
for i in range(0, nbEpisodes):
    state=env.reset()
    
    endOfEpisode = False
    while not endOfEpisode:
        actionNN.output(state)
            
        if sigma1<0.001:
            sigma1=0.001
        if sigma2<0.001:
            sigma2=0.001  
        action[0]=random.gauss(actionNN.LastLayerOutput[0],sigma1)
        action[1]=random.gauss(actionNN.LastLayerOutput[1],sigma2)
        
        next_state, reward, endOfEpisode, info = env.step(action) 
        state = next_state

        recompenseMoyenne=recompenseMoyenne+reward

recompenseMoyenne=recompenseMoyenne/nbEpisodes
print(recompenseMoyenne)

nbEpisodes=100
#evaluation random strategy
recompenseMoyenne=0
for i in range(0, nbEpisodes):
   
    state=env.reset()
    
    endOfEpisode = False
    while not endOfEpisode:
        next_state, reward, endOfEpisode, info = env.step(env.action_space.sample()) 

        state = next_state

        recompenseMoyenne=recompenseMoyenne+reward

recompenseMoyenne=recompenseMoyenne/nbEpisodes
print(recompenseMoyenne)


