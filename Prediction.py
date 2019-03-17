import tensorflow as tf
import numpy as np
import librosa
from librosa import display
import matplotlib.pyplot as plt

def conv_layer(X,weight,bias,pooling,drop_out):
    X1 = X
    if drop_out:
        X1 = tf.nn.dropout(X,keep_prob=0.75)
    out = tf.nn.conv2d(input=X1,filter=weight,strides=[1,1,1,1],padding='SAME')
    out += bias

    if pooling:
        out = tf.nn.max_pool(value=out,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    out = tf.nn.relu(out)
    return out

def fc_layer(X,weight,bias,ur,drop_out):
    X1 = X
    if drop_out:
        X1 = tf.nn.dropout(X,keep_prob=0.75)
    out= tf.matmul(X1,weight) + bias
    if ur:
        out = tf.nn.relu(out)
    return out

def get_genre(X):
    Genres = ["Blues", "Classical", "Country", "Disco",
              "HipHop", "Jazz", "Metal", "Pop", "Reggae",
              "Rock"]

    i = np.argmax(X)
    result = str(int(X[i]*100))+"% "+Genres[i]
    return result


data_raw = []
S = []

stride=1
offset=0
duration=5

pat = "genres/Examples/HipHop.mp3"
#pat = "genres/hiphop/hiphop.00000.au"

while(offset<30):
    audio,sr = librosa.load(pat,offset=offset,duration=duration)
    mfcc = librosa.feature.mfcc(y=audio,sr=sr)
    data_raw.append(mfcc)
    offset+=5

filter_size = 3
filter = [1,16,32,32]
n = [8640,1024,10]
classes = 10

bs_train = 6

data = np.reshape(np.asarray(data_raw),newshape=[bs_train,20,216,1])

graph = tf.Graph()

with graph.as_default():

    batch_train_data = tf.placeholder(tf.float32, shape=(bs_train,20,216,1), name="batch_train_data")

    F0 = tf.get_variable("F0", shape=([filter_size, filter_size, filter[0], filter[1]]))
    F1 = tf.get_variable("F1", shape=([filter_size, filter_size, filter[1], filter[2]]))
    F2 = tf.get_variable("F2", shape=([filter_size, filter_size, filter[2], filter[3]]))

    W1 = tf.get_variable("W1", shape=([n[0], n[1]]))
    W2 = tf.get_variable("W2", shape=([n[1], classes]))

    B0 = tf.get_variable("B0", shape=([filter[1]]))
    B1 = tf.get_variable("B1", shape=([filter[2]]))
    B2 = tf.get_variable("B2", shape=([filter[3]]))
    B3 = tf.get_variable("B3", shape=([n[1]]))
    B4 = tf.get_variable("B4", shape=([n[2]]))

    O1 = conv_layer(batch_train_data, F0, B0, pooling=True, drop_out=False)
    O2 = conv_layer(O1, F1, B1, pooling=True, drop_out=False)
    O3 = conv_layer(O2, F2, B2, pooling=False, drop_out=False)

    X2 = tf.reshape(O3, [bs_train, 8640])

    O4 = fc_layer(X2, W1, B3, ur=True, drop_out=False)
    Z5 = fc_layer(O4, W2, B4, ur=False, drop_out=False)
    A5 = tf.nn.softmax(Z5)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

with tf.Session(graph=graph) as sess:

    saver.restore(sess, "C:/Users/ADMIN/Desktop/MP2/MGC/model/model.ckpt")
    feed_dict = {batch_train_data: data}
    pred = sess.run(A5, feed_dict=feed_dict)
    for i in range(bs_train):
        S.append(get_genre(pred[i]))

    S.append(get_genre(pred.sum(axis=0)/bs_train))

plt.figure(figsize=(12, 4))
plt.suptitle("Overall:"+S[bs_train])

for i in range(bs_train):
    plt.subplot(1,bs_train,i+1)
    librosa.display.specshow(data=data_raw[i],x_axis='time',)
    plt.title(S[i])

plt.colorbar()
plt.show()



