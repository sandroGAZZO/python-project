from keras.models import Sequential
from keras.layers import Dense
import numpy
import operator

numpy.random.seed(1)
model = Sequential()

def getTrainedANN(model, myInput, myOutput):
	model.add(Dense(52, input_dim=42, activation='relu'))
	model.add(Dense(52, activation='relu'))
	model.add(Dense(7, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
	# Fit the model
	model.fit(myInput, myOutput, epochs=600, batch_size=30)
	# evaluate the model
	scores = model.evaluate(X, Y)

	return (model, scores)

def getPridiction(model, myInput):
	prediction = model.predict(myInput)
	index, value = max(enumerate(prediction), key=operator.itemgetter(1))
	return index + 1
