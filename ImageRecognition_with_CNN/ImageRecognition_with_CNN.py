# Importing libraries
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils
seed = 21
from keras.datasets import cifar10

#loading dataset
my_data = cifar10.load_data()
(X_train, y_train), (X_test, y_test) = my_data

#normalizing data
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train/255
X_test = X_test/255

#one hot encoding outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
class_num = y_test.shape[1]

# create the model
my_model = Sequential()
my_model.add(Conv2D(32, (3,3), input_shape=X_train.shape[1:], padding='same'))
my_model.add(Activation('relu'))
my_model.add(Dropout(0.2))
my_model.add(BatchNormalization())

my_model.add(Conv2D(64, (3,3), padding='same'))
my_model.add(Activation('relu'))
my_model.add(MaxPooling2D(pool_size=(2,2)))
my_model.add(Dropout(0.2))
my_model.add(BatchNormalization())

my_model.add(Conv2D(128, (3,3), padding='same'))
my_model.add(Activation('relu'))
my_model.add(Dropout(0.2))
my_model.add(BatchNormalization())

my_model.add(Flatten())
my_model.add(Dropout(0.2))

my_model.add(Dense(256, kernel_constraint=maxnorm(3)))
my_model.add(Activation('relu'))
my_model.add(Dropout(0.2))
my_model.add(BatchNormalization())

my_model.add(Dense(128, kernel_constraint=maxnorm(3)))
my_model.add(Activation('relu'))
my_model.add(Dropout(0.2))
my_model.add(BatchNormalization())

my_model.add(Dense(class_num))
my_model.add(Activation('softmax'))

epochs = 25
optimizer = 'adam'

my_model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics=['accuracy'])

print(my_model.summary())

np.random.seed(seed)
my_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs = epochs, batch_size = 64)

scores = my_model.evaluate(X_test, y_test, verbose=0)
print("our final accuracy is: ", scores[1]*100)