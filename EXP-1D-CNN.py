### IMPORT LIBS
import tensorflow as tf
import tensorflow.keras as keras

import random
import numpy as np
import scipy.io as scio
import pandas as pd
import os
import collections
from matplotlib import pyplot as plt

from keras import optimizers
from keras.models import Model, Sequential
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv1D, BatchNormalization, MaxPooling1D
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split


### PATH
base_path = '/home/wzx/ECGAI/EXP1/1D_CNN'

dataset_path =  '/home/wzx/ECGAI/EXP1/Dataset/' # Training data

### PLOT SETTING
plt.rcParams["figure.figsize"] = (15,5)
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.color'] = 'b'
plt.rcParams['axes.grid'] = True

### LAYERS
def nn_base(Unit, Kernel, Conv_Stride, Maxpool_Stride, input_tensor = None):
    
    input_shape = (None, 3)

    ecg_input = input_tensor

    bn_axis = 3

    # nn_base
    x = Conv1D(filters = Unit, kernel_size = Kernel, padding = 'same', strides = Conv_Stride, activation = 'relu', 
               data_format='channels_last')(ecg_input)
    
    x = BatchNormalization()(x)
    
    x = MaxPooling1D(pool_size = 2, strides = Maxpool_Stride, padding = 'same', 
                     data_format='channels_last')(x)

    return x




def classifier_layer(base_layer, dropout_rate = 0.1, ClassesNum = 17):
    
    # classifier layers
    x = Conv1D(filters = 32, kernel_size = 10, padding = 'same', activation = 'relu', 
               name = 'classifier_layer_Conv1D1', data_format='channels_last')(base_layer)
    
    x = Conv1D(filters = 128, kernel_size = 5, padding = 'same', activation = 'relu', 
               strides = 2, name = 'classifier_layer_Conv1D2', data_format='channels_last')(x)
    
    x = MaxPooling1D(pool_size = 2, strides = 2, padding = 'same', 
                     name = 'classifier_layer_MaxPooling1D1', data_format='channels_last')(x)
    
    x = Conv1D(filters = 256, kernel_size = 15, padding = 'same', activation = 'relu', 
               name = 'classifier_layer_Conv1D3', data_format='channels_last')(x)
    
    x = MaxPooling1D(pool_size = 2, strides = 2, padding = 'same', 
                     name = 'classifier_layer_MaxPooling1D2', data_format='channels_last')(x)
    
    x = Conv1D(filters = 512, kernel_size = 5, padding = 'same', activation = 'relu', 
               name = 'classifier_layer_Conv1D4', data_format='channels_last')(x)
    
    x = Conv1D(filters = 128, kernel_size = 3, padding = 'same', activation = 'relu', 
               name = 'classifier_layer_Conv1D5', data_format='channels_last')(x)
    
    
    x = Flatten(name = 'classifier_layer_Flatten')(x)
    x = Dense(units = 512, activation = 'relu', name = 'classifier_layer_Dense1')(x)
    x = Dropout(rate = dropout_rate, name = 'classifier_layer_Dropout')(x)
    x = Dense(units = ClassesNum, activation = 'softmax', name = 'classifier_layer_Dense2')(x)
    
    return x




### DATA PROCESSING
# Variables

classes = ['NSR', 'APB', 'AFL', 'AFIB', 'SVTA', 'WPW','PVC', 'Bigeminy', 'Trigeminy', 
           'VT', 'IVR', 'VFL', 'Fusion', 'LBBBB', 'RBBBB', 'SDHB', 'PR']
ClassesNum = len(classes)

X = list()
y = list()

for root, dirs, files in os.walk(dataset_path, topdown=False):
    for name in files:
        data_train = scio.loadmat(os.path.join(root, name))# 取出字典里的value
        
        # arr -> list
        data_arr = data_train.get('val')
        data_list = data_arr.tolist()
        
        X.append(data_list[0]) # [[……]] -> [ ]
        y.append(int(os.path.basename(root)[0:2]) - 1)  # name -> num


### START TRAINING


# list -> arr
X=np.array(X)
y=np.array(y)

print("total num of training data : ", len(X))

# get X_train, X_test, y_train, y_test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("X_train : ", len(X_train))
print("X_test  : ", len(X_test))
print("y_train : ", collections.Counter(y_train))
print("y_test  : ", collections.Counter(y_test))
print("shape of X_train : ", np.shape(X_train[0]))
print("shape of y_train : ", np.shape(y_train))
print("shape of X_test : ", np.shape(X_test))
print("shape of y_test : ", np.shape(y_test))


### BUILD MODEL_SEQ
'''
model_all = Sequential()
model_all.add(Conv1D(128, 50, padding = 'same', activation='relu', strides = 3,
                     input_shape=(3600,1), data_format='channels_last'))
model_all.add(BatchNormalization())
model_all.add(MaxPooling1D(3, padding = 'same', data_format='channels_last'))

model_all.add(Conv1D(32, 7, padding = 'same', activation='relu', strides = 1, data_format='channels_last'))
model_all.add(BatchNormalization())
model_all.add(MaxPooling1D(2, padding = 'same', data_format='channels_last'))

model_all.add(Conv1D(32, 10, padding = 'same', activation='relu', strides = 1, data_format='channels_last'))
model_all.add(Conv1D(128, 5, padding = 'same', activation='relu', strides = 2, data_format='channels_last'))
model_all.add(MaxPooling1D(2,padding = 'same', data_format='channels_last'))

model_all.add(Conv1D(256, 15, padding = 'same', activation='relu', strides = 1, data_format='channels_last'))
model_all.add(MaxPooling1D(2, padding = 'same', data_format='channels_last')) 

model_all.add(Conv1D(512, 5, padding = 'same', activation='relu', strides = 1, data_format='channels_last'))
model_all.add(Conv1D(128, 3, padding = 'same', activation='relu', strides = 1, data_format='channels_last'))
model_all.add(Flatten(data_format='channels_last'))

model_all.add(Dense(512, activation='relu'))
model_all.add(Dropout(0.1))
model_all.add(Dense(ClassesNum, activation='softmax'))

print(model_all.summary())
'''


### BUILD MODEL_MOD
input_ecg = Input(shape=(3600,1))
x = nn_base(128, 50, Conv_Stride = 3, Maxpool_Stride = 3, input_tensor = input_ecg)
x = nn_base(32, 7, Conv_Stride = 1, Maxpool_Stride = 2, input_tensor = x)
output_ecg = classifier_layer(x, dropout_rate = 0.1, ClassesNum = 17)
model_m = Model(input_ecg, output_ecg)
print(model_m.summary())

### SETTING OPTIMIZERS & COMPILE
# setting optimizers & compile
optimizers.Adam(lr = 0.002, beta_1 = 0.9, beta_2 = 0.999, epsilon = None, decay = 0.0, amsgrad = False)
# model_all.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model_m.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# expand X_train dims
X_train = np.expand_dims(X_train, axis=2)
X_test = np.reshape(X_test, (np.shape(X_test)[0], np.shape(X_test)[1], 1))
# Y : int -> binary (one-hot)
y_train = to_categorical(y_train,num_classes = ClassesNum)
y_test = to_categorical(y_test,num_classes = ClassesNum)

display(np.shape(X_train))


### TRAINING 

BATCH_SIZE = 16
EPOCHS = 40
# history = model_all.fit(X_train, y_train, batch_size = BATCH_SIZE, epochs = EPOCHS)
history = model_m.fit(X_train, y_train, batch_size = BATCH_SIZE, epochs = EPOCHS)


### PRINT ACC&LOSS 
print(history.history.keys())
print(history.history['loss'])
print(history.history['accuracy'])


### PRINT PLOT 
# put into df
record_arr = np.array([history.history['loss'][0], history.history['accuracy'][0]])
for i in range(1, EPOCHS):
    new_row = np.array([history.history['loss'][i], history.history['accuracy'][i]])
    record_arr = np.row_stack((record_arr, new_row))
    
record_df = pd.DataFrame(record_arr, columns=["loss", "acc"])

#print curve
plt.subplot(1,2,1)
plt.plot(np.arange(0, EPOCHS), record_df["loss"], 'r')
plt.title('loss')
plt.subplot(1,2,2)
plt.plot(np.arange(0, EPOCHS), record_df["acc"], 'r')
plt.title('acc')


### VAL ACC LOSS 
# val_loss_acc = model_all.evaluate(X_test, y_test, batch_size=100)
val_loss_acc = model_m.evaluate(X_test, y_test, batch_size = 16)
print("loss of val : ", val_loss_acc[0])
print("acc of val : ", val_loss_acc[1])

### PREDICT 

# predictions = model_all.predict(X_test)
predictions = model_m.predict(X_test)
display(predictions)



