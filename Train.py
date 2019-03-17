import tensorflow as tf
import numpy as np
import h5py
from sklearn.utils import shuffle

def new_weights(shape,nam):
    return tf.get_variable(initializer=tf.truncated_normal(stddev=0.05,shape=shape),name=nam)
    #return tf.Variable(tf.truncated_normal(shape, stddev=0.05),name=nam)

def new_biases(length,nam):
    #return tf.Variable(tf.constant(0.05, shape=[length]),name=nam)
    return tf.get_variable(initializer=tf.constant(0.05, shape=[length]),name=nam)

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

with h5py.File('genres/processed.h5','r') as hdf:
    train_data=np.array(hdf.get('train_data'))
    train_label=np.array(hdf.get('train_label'))
    test_data=np.array(hdf.get('test_data'))
    test_label=np.array(hdf.get('test_label'))

bs_train = 64
epochs = int(input('Enter Number of Epochs:'))
alpha=0.0001
stride=1
b_train = 75

Test_acc = []
Train_acc = []
Test_loss = []
Train_loss =[]

train_data = np.reshape(train_data,[4800,20,216,1])
test_data = np.reshape(test_data,[1200,20,216,1])

filter_size = 3
filter = [1,16,32,32]
n = [8640,1024,10]
classes = 10

N_train=4800
N_test=1200

graph = tf.Graph()

re = int(input('Enter 0 to start again:'))

with graph.as_default():

    batch_train_data = tf.placeholder(tf.float32, shape=(bs_train,20,216,1), name="batch_train_data")
    batch_train_label = tf.placeholder(tf.float32, shape=(bs_train,classes), name="batch_train_label")

    test_data_ph = tf.placeholder(tf.float32, shape=(N_test,20,216,1), name="test_data")
    test_label_ph = tf.placeholder(tf.float32, shape=(N_test,classes), name="test_label")

    if re == 0:
        F0 = new_weights([filter_size,filter_size,filter[0],filter[1]],"F0")
        F1 = new_weights([filter_size,filter_size,filter[1],filter[2]],"F1")
        F2 = new_weights([filter_size,filter_size,filter[2],filter[3]],"F2")

        W1 = new_weights([n[0],n[1]],"W1")
        W2 = new_weights([n[1],classes],"W2")

        B0 = new_biases(filter[1],"B0")
        B1 = new_biases(filter[2],"B1")
        B2 = new_biases(filter[3],"B2")
        B3 = new_biases(n[1],"B3")
        B4 = new_biases(n[2],"B4")
    else :
        F0 = tf.get_variable("F0", shape=([filter_size,filter_size,filter[0],filter[1]]))
        F1 = tf.get_variable("F1", shape=([filter_size,filter_size,filter[1],filter[2]]))
        F2 = tf.get_variable("F2", shape=([filter_size,filter_size,filter[2],filter[3]]))

        W1 = tf.get_variable("W1", shape=([n[0],n[1]]))
        W2 = tf.get_variable("W2", shape=([n[1],classes]))

        B0 = tf.get_variable("B0",shape=([filter[1]]))
        B1 = tf.get_variable("B1",shape=([filter[2]]))
        B2 = tf.get_variable("B2",shape=([filter[3]]))
        B3 = tf.get_variable("B3",shape=([n[1]]))
        B4 = tf.get_variable("B4",shape=([n[2]]))

    O1 = conv_layer(batch_train_data,F0,B0,pooling=True,drop_out=False)
    O2 = conv_layer(O1,F1,B1,pooling=True,drop_out=True)
    O3 = conv_layer(O2,F2,B2,pooling=False,drop_out=True)

    X2 = tf.reshape(O3,[bs_train,8640])

    O4 = fc_layer(X2,W1,B3,ur=True,drop_out=True)
    Z5 = fc_layer(O4,W2,B4,ur=False,drop_out=False)
    A5 = tf.nn.softmax(Z5)
    Y_pred = tf.argmax(A5,axis=1)
    Y_true = tf.argmax(batch_train_label,axis=1)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Z5, labels=batch_train_label))
    optimizer = tf.train.AdamOptimizer(learning_rate=alpha).minimize(loss)

    correct_prediction = tf.equal(Y_pred,Y_true)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    O1t = conv_layer(test_data_ph, F0, B0, pooling=True,drop_out=False)
    O2t = conv_layer(O1t, F1, B1, pooling=True,drop_out=False)
    O3t = conv_layer(O2t, F2, B2, pooling=False,drop_out=False)

    X2t = tf.reshape(O3t, [N_test, 8640])

    O4t = fc_layer(X2t, W1, B3, ur=True,drop_out=False)
    Z5t = fc_layer(O4t, W2, B4, ur=False,drop_out=False)
    A5t = tf.nn.softmax(Z5t)
    Y_predt = tf.argmax(A5t, axis=1)
    Y_truet = tf.argmax(test_label_ph, axis=1)
    losst = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Z5t, labels=test_label_ph))
    correct_predictiont = tf.equal(Y_predt, Y_truet)
    accuracyt = tf.reduce_mean(tf.cast(correct_predictiont, tf.float32))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

with tf.Session(graph=graph) as sess:

    if re == 0:
        sess.run(init)
    else:
        saver.restore(sess,"C:/Users/ADMIN/Desktop/MP2/MGC/model/model1.ckpt")
    for epoch in range(epochs):
        acc = 0
        los = 0

        for i in range(b_train):
            batch_data = train_data[(i * bs_train):((i + 1) * bs_train), :, :, :]
            batch_lab = train_label[(i * bs_train):((i + 1) * bs_train), :]

            feed_dict = {batch_train_data: batch_data, batch_train_label: batch_lab}

            _, B_loss,B_acc = sess.run([optimizer, loss,accuracy], feed_dict=feed_dict)

            if True: #epoch%10 == 9:
                acc+=B_acc
                los+=B_loss

        if True:#epoch%10 == 9:
            acc/=b_train
            los/=b_train
            print("***\nAT Epoch:", epoch + 1)
            print("Training set accuracy is:", acc)
            print("Training set loss is:", los)
            feed_dict = {test_data_ph: test_data, test_label_ph: test_label}
            t_acc,t_loss = sess.run([accuracyt,losst],feed_dict=feed_dict)
            print("Test set accuracy:", t_acc)
            Train_acc.append(acc)
            Train_loss.append(los)
            Test_acc.append(t_acc)
            Test_loss.append(t_loss)
        print("Epochs done:",epoch + 1)

    save_path = saver.save(sess, "C:/Users/ADMIN/Desktop/MP2/MGC/model/model1.ckpt")

Train_acc = np.asarray(Train_acc)
Test_loss = np.asarray(Test_loss)
Train_loss = np.asarray(Train_loss)
Test_acc = np.asarray(Test_acc)

with h5py.File('genres/results1.h5','w') as hdf:
    hdf.create_dataset('train_acc',data=Train_acc)
    hdf.create_dataset('test_acc', data=Test_acc)
    hdf.create_dataset('train_loss',data=Train_loss)
    hdf.create_dataset('test_loss',data=Test_loss)