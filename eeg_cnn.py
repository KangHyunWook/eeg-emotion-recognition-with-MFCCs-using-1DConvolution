#eeg cnn
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.layers import Conv1D, Activation, Flatten, Dense
from keras.models import Sequential

import pandas as pd
import librosa
import numpy as np


pd_eeg_data = pd.read_csv('emotions.csv')
eeg_data = pd_eeg_data.iloc[:, :-1]
eeg_labels = pd_eeg_data.iloc[:, -1]

eeg_data = eeg_data.values
eeg_labels = eeg_labels.values

data = []
for i in range(len(eeg_data)):
    mfccs=librosa.feature.mfcc(y=eeg_data[i], sr=150, n_mfcc=13)
    mfccs = np.mean(mfccs,axis=0)
    data.append(mfccs)
    
data=np.array(data)
y=eeg_labels

xxx = StratifiedShuffleSplit(1, test_size=0.2, random_state=12)

for train_index, test_index in xxx.split(data, y):
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = np.array(X_test)
y_test = np.array(y_test)

lb = LabelEncoder()
y_train = np_utils.to_categorical(lb.fit_transform(y_train))
y_test = np_utils.to_categorical(lb.fit_transform(y_test))

x_traincnn = np.expand_dims(X_train, axis=2)
x_testcnn = np.expand_dims(X_test, axis=2)
    
model = Sequential()
model.add(Conv1D(64, 3, padding='same', input_shape=(X_train.shape[1], 1)))
model.add(Conv1D(64, 3, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(32, 1))
model.add(Conv1D(32, 1))
model.add(Flatten())
model.add(Dense(3))
model.add(Activation('softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_traincnn, y_train, validation_data=(x_testcnn, y_test), epochs=30, batch_size=20)






























