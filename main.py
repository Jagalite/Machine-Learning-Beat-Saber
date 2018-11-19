from pydub import AudioSegment
import json
import numpy as np

from scipy import signal
from scipy.io import wavfile

from keras.models import Sequential
from keras.layers import Dense

np.random.seed(7)

songsData = []
songsLabel = []

def readSongAndEvents(songPath, eventsPath):
    
    global songsData
    song = AudioSegment.from_ogg('Training/Beat it/Beat It.ogg').get_array_of_samples()
    inputData = np.asarray( song, dtype="int32" )
    songsData.append(inputData)
    
    with open('Training/Beat it/Easy.json', 'r') as jsonFile:
        
        #load events
        jsonObj = json.loads(jsonFile.read())
        eventsDict = jsonObj['_events']
        
        eventsArray = []
        for event in eventsDict:
            eventsArray.append(event['_time'])
        
        outputData = np.asarray( eventsArray, dtype="int32" )
        
        global songsLabel
        songsLabel.append(outputData)
        
readSongAndEvents('Training/Beat it/Beat It.ogg', 'Training/Beat it/Easy.json')
readSongAndEvents('Training/Beat it/Beat It.ogg', 'Training/Beat it/Medium.json')
readSongAndEvents('Training/Beat it/Beat It.ogg', 'Training/Beat it/Hard.json')
readSongAndEvents('Training/Beat it/Beat It.ogg', 'Training/Beat it/Expert.json')
    
songsData = np.asarray(songsData)
songsLabel = np.asarray(songsLabel)

print(len(songsData))
print(len(songsLabel))

model = Sequential()
model.add(Dense(units=13, activation='relu', input_shape=songsData[0].shape))
model.add(Dense(units=878, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd',metrics=['accuracy'])
model.fit(songsData, songsLabel, epochs=5, batch_size=10)
