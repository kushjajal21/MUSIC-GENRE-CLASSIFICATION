import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
plt.rcParams['figure.figsize'] = (12,12)

def lab(i):
    label=[]
    for k in range(10):
        if k==i:
            label.append(1)
        else:
            label.append(0)
    print(label)
y, s = librosa.load("genres/metal/metal.00000.au",duration=5)
y2, s2  = librosa.load("genres/jazz/jazz.00000.au",duration=5)

mel = librosa.feature.melspectrogram(y=y,sr=s)
mel2 = librosa.feature.melspectrogram(y=y2,sr=s2)
mfcc = librosa.feature.mfcc(y=y,sr=s)
mfcc2 = librosa.feature.mfcc(y=y2,sr=s2)

plt.figure(figsize=(12, 4))

plt.suptitle("MFCCs:")

plt.subplot(1,2,1)
plt.title("Metal.00000")
librosa.display.specshow(mfcc,x_axis="time")
plt.colorbar()


plt.subplot(1,2,2)
plt.title("Jazz.00000")
librosa.display.specshow(mfcc2,x_axis="time")
plt.colorbar()
plt.show()