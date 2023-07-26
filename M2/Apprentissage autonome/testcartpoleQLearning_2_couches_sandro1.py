#FrozenLake with "Deep (not so deep) Q-learning"

import gym
import random
import numpy
import math

env = gym.make('CartPole-v0')

epsilon=0.1
gamma=0.9
alpha=0.5



Q = numpy.zeros([env.observation_space.shape[0], env.action_space.n])

def sigmoid(x):
    if x>100:
        returnValue=1
    elif x<-100:
        returnValue=-1
    else:
        returnValue=(math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x))
    return returnValue
        
class NN:
    def __init__(self,sizeInput,sizeHiddenLayer1,sizeHiddenLayer2,sizeOutput):
        self.sizeInput=sizeInput
        self.sizeHiddenLayer1=sizeHiddenLayer1
        self.sizeHiddenLayer2=sizeHiddenLayer2
        self.sizeOutput=sizeOutput

        #below are the weights
        self.HiddenLayer1EntryWeights=numpy.zeros([sizeHiddenLayer1,sizeInput])
        self.HiddenLayer2EntryWeights=numpy.zeros([sizeHiddenLayer2,sizeHiddenLayer1])
        self.LastLayerEntryWeights=numpy.zeros([sizeOutput,sizeHiddenLayer2])

        #random initialization
        for i in range(0,sizeHiddenLayer1):
            for j in range(0,sizeInput):
                self.HiddenLayer1EntryWeights[i,j]=random.uniform(-0.1,0.1)
        
        for i in range(0,sizeHiddenLayer2):
            for j in range(0,sizeHiddenLayer1):
                self.HiddenLayer2EntryWeights[i,j]=random.uniform(-0.1,0.1)
                
        for i in range(0,sizeOutput):
            for j in range(0,sizeHiddenLayer2):
                self.LastLayerEntryWeights[i,j]=random.uniform(-0.1,0.1)

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
            self.LastLayerOutput[i]= \
            sigmoid(numpy.dot(self.LastLayerEntryWeights[i],self.HiddenLayer2Output))

    def retropropagation(self,x,y,actionIndex):
        self.output(x)

        #deltas computation
        self.LastLayerEntryDeltas[actionIndex]=2*(self.LastLayerOutput[actionIndex]-y)* \
            (1+self.LastLayerOutput[actionIndex])*(1-self.LastLayerOutput[actionIndex])

        for i in range(0,self.sizeHiddenLayer2):
            self.HiddenLayer2EntryDeltas[i]=self.LastLayerEntryDeltas[actionIndex]* \
            (1+self.HiddenLayer2Output[i])*(1-self.HiddenLayer2Output[i])*self.LastLayerEntryWeights[actionIndex,i]
            
            
        for i in range(0,self.sizeHiddenLayer1):
            self.HiddenLayer1EntryDeltas[i]=0
            for j in range(0,self.sizeHiddenLayer2):
                self.HiddenLayer1EntryDeltas[i]+=self.HiddenLayer2EntryDeltas[j]* \
                (1+self.HiddenLayer1Output[i])*(1-self.HiddenLayer1Output[i])*self.HiddenLayer2EntryWeights[j,i]
            
        #weights update
        for i in range(0,self.sizeHiddenLayer2):
            self.LastLayerEntryWeights[actionIndex,i]-=alpha*self.LastLayerEntryDeltas[actionIndex]* \
            self.HiddenLayer2Output[i]

        for i in range(0,self.sizeHiddenLayer2):
            for j in range(0,self.sizeHiddenLayer1):
                self.HiddenLayer2EntryWeights[i,j]-=alpha*self.HiddenLayer2EntryDeltas[i]*self.HiddenLayer1Output[j]
                
        for i in range(0,self.sizeHiddenLayer1):
            for j in range(0,self.sizeInput):
                self.HiddenLayer1EntryWeights[i,j]-=alpha*self.HiddenLayer1EntryDeltas[i]*x[j]
nbCells=4
sizeInput=nbCells
sizeHiddenLayer1=10
sizeHiddenLayer2=10
sizeOutput=2
myNN = NN(sizeInput, sizeHiddenLayer1,sizeHiddenLayer2, sizeOutput)

nbEpisodes=50000
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
        successesInARow=0

    state=env.reset()
    
    firstState=state
    endOfEpisode = False
    nbSteps=0

    while not endOfEpisode:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            x=state
            myNN.output(x)
            action = numpy.argmax(myNN.LastLayerOutput)
            
        next_state, reward, endOfEpisode, info = env.step(action) 
      
        myNN.output(x)
        next_max = numpy.max(myNN.LastLayerOutput)
        
        if endOfEpisode==True:
            reward=0

        target = reward+gamma*next_max


#        if reward==1:
#            successesInARow=successesInARow+1
#            print("successesInARow: "+str(successesInARow))

        myNN.retropropagation(x,target,action)
        
        state = next_state
        nbSteps=nbSteps+1
print("end of learning period")

#evaluation
averageNumberSuccesses=0

nbEpisodes=10000
for i in range(0, nbEpisodes):
    if i%100==0:
        print("episode: "+str(i))
    state=env.reset()
    
    endOfEpisode = False
    while not endOfEpisode:
        x=numpy.zeros(sizeInput)
        x[state-1]=1
        myNN.output(x)
        action = numpy.argmax(myNN.LastLayerOutput)
        
        next_state, reward, endOfEpisode, info = env.step(action) 

        state = next_state
        averageNumberSuccesses=averageNumberSuccesses+1

averageNumberSuccesses=averageNumberSuccesses/nbEpisodes

print(averageNumberSuccesses)



#evaluation random strategy
averageNumberSuccesses=0
for i in range(0, nbEpisodes):
   
    state=env.reset()
    
    endOfEpisode = False
    while not endOfEpisode:
        next_state, reward, endOfEpisode, info = env.step(env.action_space.sample()) 

        state = next_state

    if reward==1:
        averageNumberSuccesses=averageNumberSuccesses+1
averageNumberSuccesses=averageNumberSuccesses/nbEpisodes

print(averageNumberSuccesses)



