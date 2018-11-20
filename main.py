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
        if(round(times[i]) not in timeToSpecMap):
            timeToSpecMap[round(times[i])] = []
        freq = []
        for j in range(len(frequencies)):
            freq.append(spectrogram[j][i])
        timeToSpecMap[round(times[i])].append(freq)

    
    # songsData.append(frequencies)
    
    with open('Training/Beat it/Easy.json', 'r') as jsonFile:
        
        #load events
        jsonObj = json.loads(jsonFile.read())
        notesDict = jsonObj['_notes']
        
        notesSet = set()
        for note in notesDict:
            notesSet.add(round(note['_time']))
        
        global songsData
        global songsLabel
        
        for time in timeToSpecMap.keys():
            beat = 0
            if time in notesSet:
                beat = 1
            for freq in timeToSpecMap[time]:
                songsData.append(freq)
                songsLabel.append(beat)
        
        #outputData = np.asarray( eventsArray, dtype="int32" )
        #songsLabel.append(outputData)
        
# readSongAndEvents('Training/Beat it/Beat It.ogg', 'Training/Beat it/Easy.json')
# readSongAndEvents('Training/Beat it/Beat It.ogg', 'Training/Beat it/Medium.json')
# readSongAndEvents('Training/Beat it/Beat It.ogg', 'Training/Beat it/Hard.json')
readSongAndEvents('Training/Beat it/Beat It.ogg', 'Training/Beat it/Expert.json')

readSongAndEvents("Training/Livin' On A Prayer/song.ogg", 'Training/Beat it/Expert.json')
    
songsData = np.asarray(songsData)

songsLabel = np.asarray(songsLabel)

songsLabel = to_categorical(songsLabel)

print(songsData.shape)
print(songsLabel.shape)

model = Sequential()
model.add(Dense(units=10, activation='relu', input_shape=songsData[0].shape))
model.add(Dense(units=2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd',metrics=['accuracy'])
model.fit(songsData, songsLabel, epochs=10, batch_size=10)


song = AudioSegment.from_ogg("Training/Livin' On A Prayer/song.ogg")
song = song.set_channels(1)
song.export("Training/Livin' On A Prayer/song.wav", format="wav")
sample_rate, samples = wavfile.read("Training/Livin' On A Prayer/song.wav")
frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

newData = []
for i in range(len(times)):
    freq = []
    for j in range(len(frequencies)):
        freq.append(spectrogram[j][i])
    newData.append(freq)

newData = np.asarray(newData)

prediction = model.predict(newData)

print(prediction)

songJson = {}
songJson["_version"] = "1"
songJson["_beatsPerMinute"] = 120
songJson["_beatsPerBar"] = 16
songJson["_noteJumpSpeed"] = 10
songJson["_shuffle"] = 0
songJson["_shufflePeriod"] = 0.5
songJson["_events"] = []
songJson["_notes"] = []
songJson["_obstacles"] = []

timeCount = 3
for i in range(210, len(prediction), 35):
    if prediction[i][0] < prediction[i][1]:
        timeCount += 0.5
        note = {}
        note["_time"] = timeCount
        note["_lineIndex"] = random.randint(2,3)
        note["_lineLayer"] = random.randint(0,2)
        note["_type"] = 1
        note["_cutDirection"] = 8
        songJson["_notes"].append(note)
        
with open("Results/Livin' On A Prayer/Expert.json", 'w') as outfile:
    json.dump(songJson, outfile)
    
