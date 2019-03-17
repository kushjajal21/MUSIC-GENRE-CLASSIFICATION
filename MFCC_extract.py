import librosa
import numpy as np
import h5py
import random

def getstring(i):
    s="0"
    if i<10:
        s= s + str(i);
    else:
        s=""
        s=s + str(i)
    return s

def lab(i):
    label=[]
    for k in range(10):
        if k==i:
            label.append(1)
        else:
            label.append(0)
    return label

Genres = ["blues/blues.000","classical/classical.000","country/country.000","disco/disco.000","hiphop/hiphop.000","jazz/jazz.000","metal/metal.000","pop/pop.000","reggae/reggae.000","rock/rock.000"]

train_per = 90
test_per = 10

train_data = []
train_label = []
test_data = []
test_label = []

for gen in Genres:
    print(gen)
    for i in range(train_per):
        offset=0
        duration=5
        pat = "genres/" + gen + getstring(i) + ".au"
        while(offset<30):
            audio,sr = librosa.load(pat,offset=offset,duration=duration)
            mfcc = librosa.feature.mfcc(y=audio,sr=sr)
            train_data.append(mfcc)
            print(np.asarray(train_data).shape)
            train_label.append(lab(Genres.index(gen)))
            offset+=5
        print(i)

train_data = np.asarray(train_data)

for gen in Genres:
    print(gen)
    for i in range(train_per,train_per+test_per):
        offset=0
        duration=5
        pat = "genres/" + gen + getstring(i) + ".au"
        while(offset<30):
            audio,sr = librosa.load(pat,offset=offset,duration=duration)
            mfcc = librosa.feature.mfcc(y=audio,sr=sr)
            test_data.append(mfcc)
            test_label.append(lab(Genres.index(gen)))
            offset+=5
        print(i)


train_label = np.asarray(train_label)
test_data = np.asarray(test_data)
test_label = np.asarray(test_label)
print(train_data.shape,train_label.shape,test_data.shape,test_label.shape)

with h5py.File('genres/extracted.h5','w') as hdf:
    hdf.create_dataset('train_data',data=train_data)
    hdf.create_dataset('test_data', data=test_data)
    hdf.create_dataset('train_label',data=train_label)
    hdf.create_dataset('test_label',data=test_label)