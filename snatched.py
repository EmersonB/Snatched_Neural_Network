from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.models import model_from_json
import numpy as np
import pandas as pd

model = Sequential()

def train(file):
    model.add(Dense(10, input_dim=12, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    training_data = pd.read_csv(file)
    train_x = training_data.iloc[:, 0:12].values
    train_y = training_data.iloc[:, 12].values
    encoding_train_y = np_utils.to_categorical(train_y)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_x, encoding_train_y, epochs=300, batch_size=10)
    return model

def test(file):
    test_data = pd.read_csv(file)
    test_x = test_data.iloc[:, 0:12].values
    test_y = test_data.iloc[:, 12].values
    encoding_test_y = np_utils.to_categorical(test_y)
    scores = model.evaluate(test_x, encoding_test_y)
    print("\nAccuracy: %.2f%%" % (scores[1]*100))

def serialize_model(modelname):
    # serialize model to JSON
    model_json = model.to_json()
    with open("./model/"+modelname+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("./model/"+modelname+".h5", overwrite=True)
    print("Saved model to disk")

def load_and_test(filename, modelname):
    json_file = open("./model/"+modelname+".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("./model/"+modelname+".h5")
    print("Loaded model from disk")

    test_data = pd.read_csv(filename)
    test_x = test_data.iloc[:, 0:12].values
    test_y = test_data.iloc[:, 12].values
    encoding_test_y = np_utils.to_categorical(test_y)
    # evaluate loaded model on test data
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    score = model.evaluate(test_x, encoding_test_y, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))

# train('./train/GzTrain.csv')
# test('./test/GzTest.csv')

# train('./train/GzTrain.csv')
# serialize_model("Gz")

# load_and_test('./test/GzTest.csv', "Gz")