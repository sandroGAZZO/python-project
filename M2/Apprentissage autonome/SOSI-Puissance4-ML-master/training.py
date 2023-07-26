from create_bd import *
from reseau_neurones import *

import numpy as np
np.random.seed(42)
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# X=[]
# target=[]
# (X,target)=extraireInputEtTarget()
#
# XtoTrain=X[:301]
# targetToTrain=target[:301]
# XtoTest=X[300:]
# targetToTest=target[300:]



dataset = pd.read_csv("data/connect-4-target.data", header=None)

X = dataset.iloc[:,:42]

# Encodage des string en int
for column in X.columns:
    if X[column].dtype == type(object):
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])

Y = dataset.iloc[:,42:]
Y = np_utils.to_categorical(Y)

X = np.array(X)
Y = np.array(Y)



model = Sequential()
model.add(Dense(32, input_dim=42, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X[1:57000:2,:], Y[1:57000:2,:], epochs=150, batch_size=50)
print("training done")
model.save("data/keras_model")
scores = model.evaluate(X[2:57000:2,:], Y[2:57000:2,:])
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


while True:
    n = int(input())
    for k in range(n,n+50):
        yy = model.predict(np.reshape(X[k,:],(1,42)), batch_size=1, verbose=0)

        index, value = max(enumerate(yy[0]), key=operator.itemgetter(1))
        indexa, value = max(enumerate(Y[k,:]), key=operator.itemgetter(1))

        print(index," ===> ",indexa," oui " if index==indexa else " non non non !")
