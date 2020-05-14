import os
import time

os.environ['TF_DISABLE_MKL'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

import ctypes
from ctypes import CDLL
import matplotlib.pyplot as plt
import numpy as np
import csv
import tensorflow.keras as keras
from sklearn.metrics import confusion_matrix


def load_iris_dataset():
    lines = []
    with open('iris.data') as data_file:
        reader = csv.reader(data_file)
        for line in reader:
            if len(line) == 0:
                continue
            lines.append(line)
    dataset_inputs = np.zeros((len(lines), 4))
    dataset_expected_outputs = np.zeros((len(lines), 3))
    for i, line in enumerate(lines):
        dataset_inputs[i] = np.array([float(col) for col in line[:4]])
        if line[4] == 'Iris-setosa':
            dataset_expected_outputs[i] = np.array([1, -1, -1])
        elif line[4] == 'Iris-versicolor':
            dataset_expected_outputs[i] = np.array([-1, 1, -1])
        else:
            dataset_expected_outputs[i] = np.array([-1, -1, 1])
    print(dataset_inputs.shape)
    print(dataset_expected_outputs.shape)
    split_indexes = np.arange(len(dataset_inputs))
    np.random.shuffle(split_indexes)
    train_size = int(np.floor(len(dataset_inputs) * 0.7))
    x_train = dataset_inputs[split_indexes][:train_size]
    x_test = dataset_inputs[split_indexes][train_size:]
    y_train = dataset_expected_outputs[split_indexes][:train_size]
    y_test = dataset_expected_outputs[split_indexes][train_size:]

    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    path_to_dll = "./Machine_Learning_Lib.dll"
    cpp_lib = CDLL(path_to_dll)

    #print(cpp_lib.test_sum(5, 4))

    cpp_lib.linear_model_create.argtypes = [ctypes.c_int]
    cpp_lib.linear_model_create.restype = ctypes.c_void_p

    cpp_lib.linear_model_predict_regression.argtypes = [ctypes.c_void_p,
                                                        ctypes.c_double * 1,
                                                        ctypes.c_int]
    cpp_lib.linear_model_predict_regression.restype = ctypes.c_double

    cpp_lib.linear_model_train_classification.argtypes = [ctypes.c_void_p,
                                                          ctypes.c_double * 1,
                                                          ctypes.c_int]
    cpp_lib.linear_model_train_classification.restype = ctypes.c_double

    model = cpp_lib.linear_model_create(1)

    inputs = np.arange(10)
    predicted_values = np.zeros(10)

    for i in range(10):
        predicted_values[i] = cpp_lib.linear_model_predict_regression(model, (ctypes.c_double * 1)(*[inputs[i]]), 1)

    plt.scatter(inputs, predicted_values)
    #plt.show()

    (x_train, y_train), (x_test, y_test) = load_iris_dataset()

    epochs = 500
    alpha = 0.01

    # Test Keras Model
    # model = keras.models.Sequential()
    # model.add(keras.layers.Dense(16, activation=keras.activations.tanh))
    # model.add(keras.layers.Dense(3, activation=keras.activations.tanh))
    # model.compile(keras.optimizers.SGD(alpha), loss=keras.losses.mean_squared_error)
    #
    # start_time = time.time()
    #
    # model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=1, verbose=0)
    #
    # print(f'It took {time.time() - start_time} seconds to train in Keras')
    #
    # good_classified_on_train = 0
    # for k in range(len(x_train)):
    #     if np.argmax(model.predict(np.array([x_train[k]]))) == np.argmax(y_train[k]):
    #         good_classified_on_train += 1
    #
    # good_classified_on_test = 0
    # for k in range(len(x_test)):
    #     if np.argmax(model.predict(np.array([x_test[k]]))) == np.argmax(y_test[k]):
    #         good_classified_on_test += 1
    #
    # print(f"Keras Accuracy on train : {good_classified_on_train / len(x_train) * 100}%")
    # print(f"Keras Accuracy on test : {good_classified_on_test / len(x_test) * 100}%")
    #
    # predicted_values_on_test = model.predict(x_test)
    # predicted_values_on_test = np.argmax(predicted_values_on_test, axis=1)
    #
    # expected_values_on_test = np.argmax(y_test, axis=1)
    #
    # print(f"Confusion matrix of keras model :")
    # print(confusion_matrix(expected_values_on_test, predicted_values_on_test))
    #
    # print(f"----------------------------------------------------------------")

    # Test MyLib Model
    cpp_lib.mlp_model_create.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int
    ]
    cpp_lib.mlp_model_create.restype = ctypes.c_void_p

    cpp_lib.mlp_model_predict_classification.argtypes = [ctypes.c_void_p,
                                                         ctypes.POINTER(ctypes.c_double)]
    cpp_lib.mlp_model_predict_classification.restype = ctypes.POINTER(ctypes.c_double)

    cpp_lib.mlp_model_train_classification.argtypes = [ctypes.c_void_p,
                                                       ctypes.POINTER(ctypes.c_double),
                                                       ctypes.c_int,
                                                       ctypes.c_int,
                                                       ctypes.POINTER(ctypes.c_double),
                                                       ctypes.c_int,
                                                       ctypes.c_int,
                                                       ctypes.c_double]
    cpp_lib.mlp_model_train_classification.restype = None

    npl = np.array([len(x_train[0]), 16, len(y_train[0])])
    my_model = cpp_lib.mlp_model_create(npl.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), len(npl))
    flattened_x_train = np.reshape(x_train, (len(x_train) * len(x_train[0])))
    flattened_y_train = np.reshape(y_train, (len(y_train) * len(y_train[0])))

    start_time = time.time()
    print(alpha)
    cpp_lib.mlp_model_train_classification(my_model,
                                           flattened_x_train.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                           len(x_train),
                                           len(x_train[0]),
                                           flattened_y_train.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                           len(y_train[0]),
                                           epochs * len(x_train),
                                           alpha
                                           )

    print(f'It took {time.time() - start_time} seconds to train in our lib')

    good_classified_on_train = 0.0
    for k in range(len(x_train)):
        rslt = np.zeros(len(y_train[0]))

        rslt_pointer = cpp_lib.mlp_model_predict_classification(my_model, x_train[k].ctypes.data_as(
            ctypes.POINTER(ctypes.c_double)))
        for i in range(len(rslt)):
            rslt[i] = rslt_pointer[i]

        if np.argmax(rslt) == np.argmax(y_train[k]):
            good_classified_on_train += 1

    good_classified_on_test = 0.0

    predicted_values_on_test = np.zeros(len(x_test))

    for k in range(len(x_test)):
        rslt = np.zeros(len(y_test[0]))

        rslt_pointer = cpp_lib.mlp_model_predict_classification(my_model, x_test[k].ctypes.data_as(
            ctypes.POINTER(ctypes.c_double)))
        for i in range(len(rslt)):
            rslt[i] = rslt_pointer[i]

        predicted_values_on_test[k] = np.argmax(rslt)
        if np.argmax(rslt) == np.argmax(y_test[k]):
            good_classified_on_test += 1

    print(f"MyModel Accuracy on train : {good_classified_on_train / len(x_train) * 100}%")
    print(f"MyModel Accuracy on test : {good_classified_on_test / len(x_test) * 100}%")

    expected_values_on_test = np.argmax(y_test, axis=1)
    print(f"Confusion matrix of our model :")
    print(confusion_matrix(expected_values_on_test, predicted_values_on_test))