from pydub import AudioSegment
import json
import numpy as np

from scipy import signal
from scipy.io import wavfile

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

np.random.seed(7)

songsData = []
songsLabel = []

def readSongAndEvents(songPath, eventsPath):
    
    # global songsData
    # song = AudioSegment.from_ogg('Training/Beat it/Beat It.ogg').get_array_of_samples()
    # inputData = np.asarray( song, dtype="int32" )
    
    song = AudioSegment.from_ogg("Training/Beat it/Beat It.ogg")
    song = song.set_channels(1)
    song.export("Training/Beat it/Beat It.wav", format="wav")
    sample_rate, samples = wavfile.read('Training/Beat it/Beat It.wav')
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
    timeToSpecMap = {}
    for i in range(len(times)):
        if(round(times[i],1) not in timeToSpecMap):
            timeToSpecMap[round(times[i],1)] = []
        freq = []
        for j in range(len(frequencies)):
            freq.append(spectrogram[j][i])
        timeToSpecMap[round(times[i],1)].append(freq)

    
    # songsData.append(frequencies)
    
    with open('Training/Beat it/Easy.json', 'r') as jsonFile:
        
        #load events
        jsonObj = json.loads(jsonFile.read())
        eventsDict = jsonObj['_events']
        
        eventsArray = []
        for event in eventsDict:
            eventsArray.append(round(event['_time'],1))
        eventsSet = set(eventsArray)
        
        global songsData
        global songsLabel
        
        for time in timeToSpecMap.keys():
            beat = 0
            if time in eventsArray:
                beat = 1
            for freq in timeToSpecMap[time]:
                songsData.append(freq)
                songsLabel.append(beat)
        
        #outputData = np.asarray( eventsArray, dtype="int32" )
        #songsLabel.append(outputData)
        
readSongAndEvents('Training/Beat it/Beat It.ogg', 'Training/Beat it/Easy.json')
readSongAndEvents('Training/Beat it/Beat It.ogg', 'Training/Beat it/Medium.json')
readSongAndEvents('Training/Beat it/Beat It.ogg', 'Training/Beat it/Hard.json')
readSongAndEvents('Training/Beat it/Beat It.ogg', 'Training/Beat it/Expert.json')
    
songsData = np.asarray(songsData)

songsLabel = np.asarray(songsLabel)
songsLabel = to_categorical(songsLabel)

print(songsData)
print(songsLabel)

model = Sequential()
model.add(Dense(units=10, activation='relu', input_shape=songsData[0].shape))
model.add(Dense(units=2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd',metrics=['accuracy'])
model.fit(songsData, songsLabel, epochs=5, batch_size=10)


