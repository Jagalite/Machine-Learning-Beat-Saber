from pydub import AudioSegment
import json
import numpy as np

from keras.models import Sequential
from keras.layers import Dense

np.random.seed(7)

songsData = []
songsLabel = []

song = AudioSegment.from_ogg("Training/Beat it/Beat It.ogg").get_array_of_samples()
inputData = np.asarray( song, dtype="int32" )

with open('Training/Beat it/Easy.json', 'r') as jsonFile:
    
    #load song
    song = AudioSegment.from_ogg("Training/Beat it/Beat It.ogg").get_array_of_samples()
    inputData = np.asarray( song, dtype="int32" )
    
    #load events
    jsonObj = json.loads(jsonFile.read())
    eventsDict = jsonObj['_events']
    
    eventsArray = []
    for event in eventsDict:
        #eventsArray.append(list(event.values()))
        eventsArray.append(event['_time'])
    
    outputData = np.asarray( eventsArray, dtype="int32" )
    
    songsData.append(inputData)
    songsLabel.append(outputData)
    
    print(inputData.shape)
    print(outputData.shape)
    
songsData = np.asarray(songsData)
songsLabel = np.asarray(songsLabel)

model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=12624768))
model.add(Dense(units=878, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd',metrics=['accuracy'])
model.fit(songsData, songsLabel, epochs=1, batch_size=1)
