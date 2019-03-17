from sklearn.utils import shuffle
import numpy as np
import h5py

with h5py.File('genres/extracted.h5','r') as hdf:
    train_data=np.array(hdf.get('train_data'))
    train_label=np.array(hdf.get('train_label'))
    test_data=np.array(hdf.get('test_data'))
    test_label=np.array(hdf.get('test_label'))

    data = np.concatenate((train_data,test_data),axis=0)
    label = np.concatenate((train_label,test_label),axis=0)
    print(data.shape,label.shape)

    data,label = shuffle(data,label)

    train_data = data[0:4800,:,:]
    train_label = label[0:4800,:]
    test_label = label[4800:6000,:]
    test_data = data[4800:6000,:,:]
    print(test_data.shape,train_data.shape,train_label.shape,test_label.shape)
    print(test_label[120],test_label[122],train_label[222],train_label[225])

with h5py.File('genres/processed.h5','w') as hdf:
    hdf.create_dataset('train_data',data=train_data)
    hdf.create_dataset('test_data', data=test_data)
    hdf.create_dataset('train_label',data=train_label)
    hdf.create_dataset('test_label',data=test_label)