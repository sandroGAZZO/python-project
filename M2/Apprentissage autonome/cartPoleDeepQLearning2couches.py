import gym
import random
import numpy
import math

env = gym.make("CartPole-v1")
print(env.observation_space)
print(env.action_space)


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

nbCells=4
sizeInput=nbCells
sizeHiddenLayer1=20
sizeHiddenLayer2=20
sizeOutput=2
myNN = NN(sizeInput, sizeHiddenLayer1, sizeHiddenLayer2, sizeOutput,["softplus","softplus"])

nbEpisodes=2000
totalNbSteps=0
tailleFenetreGlissante=50
derniersScores=[-200]*tailleFenetreGlissante
fin=False

for i in range(0, nbEpisodes):
    epsilon=epsilon*epsilonDecay
    #if i>5000:
    #    alpha=0.1
    #if i>10000:
    #    alpha=0.05
    #if i>20000:
    #    alpha=0.02
    #f i>30000:
     #   alpha=0.01
    
#    if i%100==0 and i>0:
    if i>0:
        print("episode: "+str(i))
        print(totalNbSteps/i)

    state=env.reset()
    
    firstState=state
    endOfEpisode = False

    nbSteps=0
    while not endOfEpisode:
        #calcul de la moyenne glissante
        moyenneGlissante=0
        for j in range(tailleFenetreGlissante-10,tailleFenetreGlissante):
            moyenneGlissante+=derniersScores[j]
        moyenneGlissante/=10

        if moyenneGlissante>150:
            env.render()

        
        moyenneGlissante=0
        for j in range(0,tailleFenetreGlissante):
            moyenneGlissante+=derniersScores[j]
        moyenneGlissante/=tailleFenetreGlissante

        if moyenneGlissante>400:
            fin=True
            break
        
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            x=numpy.zeros(sizeInput)
            x=state
            myNN.output(x)
            action = numpy.argmax(myNN.LastLayerOutput)
            
        next_state, reward, endOfEpisode, info = env.step(action) 

        if endOfEpisode:
          reward=0
        
        x=next_state
        myNN.output(x)
        next_max = numpy.max(myNN.LastLayerOutput)

        target = reward+gamma*next_max

        x=numpy.zeros(sizeInput)
        x=state

        myNN.retropropagation(x,target,action,alpha)
        
        state = next_state
        nbSteps=nbSteps+1

    if i<tailleFenetreGlissante:
        derniersScores[i]=nbSteps
    else:
        derniersScores.pop(0)
        derniersScores.append(nbSteps)

    totalNbSteps=totalNbSteps+nbSteps
        
    if fin==True:
        break
print("end of learning period")


#evaluation
averageLength=0

nbEpisodes=1000
for i in range(0, nbEpisodes):
    state=env.reset()
    
    endOfEpisode = False
    while not endOfEpisode:
        x=state
        myNN.output(x)
        action = numpy.argmax(myNN.LastLayerOutput)
        
        next_state, reward, endOfEpisode, info = env.step(action) 

        state = next_state

        if reward==1:
          averageLength=averageLength+1

averageLength=averageLength/nbEpisodes

print(averageLength)


averageLength=0
#evaluation random strategy
for i in range(0, nbEpisodes):
   
    state=env.reset()
    
    endOfEpisode = False
    while not endOfEpisode:
        next_state, reward, endOfEpisode, info = env.step(env.action_space.sample()) 

        state = next_state

        if reward==1:
          averageLength=averageLength+1
averageLength=averageLength/nbEpisodes

print(averageLength)


