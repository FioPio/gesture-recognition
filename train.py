#!/usr/bin/python3
# -*- coding: utf-8 -*-

DATA_FILE = 'training_data/data'
NCLASSES  = 4

################################################################################
#
#                                LOADING DATA
#
################################################################################
import numpy as np

def loadData(classnumber):
    '''
        This function loads the training data for a class.

        =======
        PARAMS
        =======

         classnumber : int
              The number that represents the class.

        =======
        RETURNS
        =======

         labels      : list of int
              A list containing the class of the elements loaded.

         data        : list of floats
              A list containing the recorded data for that class.
    '''
    print(f'[INFO] Loading data for class {classnumber}')
    # Reading the file
    with open(DATA_FILE+str(classnumber)+'.txt', 'r') as f:
        bkgd = f.readlines()
        f.close()

    # Removing unnecessary symbols
    lines = [line.replace('[', '').replace(']',',').replace('\n','').split(',') for line in bkgd]

    # Getting the individual strings of just floats
    data = []
    labels = []
    label = np.zeros(NCLASSES)
    label[classnumber] = 1.0
    for  idx, line in enumerate(lines):
        data.append([])
        for val in line:
            data[idx].append(val)
        data[idx] = list(filter(None, data[idx]))
        # Creates an array with class label
        labels.append(label)

    # Converting them to floats
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j]=float(data[i][j])
    print(f'[INFO] Loaded {len(labels)} elements')
    return data, labels

y = []
X = []

# Loading the data
for i in range(NCLASSES):
    tX, ty = loadData(i)
    y.extend(ty)
    X.extend(tX)

################################################################################
#
#               SPLITTING DATA INTO TRAIN AND TEST SETS
#
################################################################################
from sklearn.model_selection import train_test_split

RANDOM_SEED = 32

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.75, random_state=RANDOM_SEED)

################################################################################
#
#                            BUILDING THE MODEL
#
################################################################################
import tensorflow as tf

MODEL = 'models/modelv2.hdf5'

model = tf.keras.models.Sequential([
    tf.keras.layers.Input((len(X_train[0]), )),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(800, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(400, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(NCLASSES, activation='softmax')
])

# Show the summary of the model
model.summary()

# What to save
cp_callback = tf.keras.callbacks.ModelCheckpoint( MODEL,
                                                  verbose=1,
                                                  save_weights_only=False
                                                 )
# Early stop
es_callback = tf.keras.callbacks.EarlyStopping(patience = 20, verbose=1)
model.compile( optimizer = 'adam',
               loss='categorical_crossentropy',
               metrics=['accuracy']
              )

################################################################################
#
#                            TRAINING THE MODEL
#
################################################################################

model.fit( np.array(X_train),
           np.array(y_train),
           epochs=1000,
           batch_size=128,
           validation_data=(np.array(X_test),np.array(y_test)),
           callbacks=[cp_callback, es_callback]
           )
