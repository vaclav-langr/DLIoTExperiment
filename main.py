import pandas as pd
import numpy as np
import ipaddress
import datetime
import os
import json
import sklearn.preprocessing as preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import keras
from enum import Enum

DATA_FOLDER = 'E:\\Downloads\\Experiment\\'
LABEL_ENCODER = None


class Model(Enum):
    MLP = 'MLP'
    CNN = 'CNN'
    LSTM = 'LSTM'
    CNN_LSTM = 'CNN_LSTM'


def split_data(data, labels, train_size_percent=0.8, shuffle=False):
    if shuffle:
        p = np.random.permutation(len(data))
        data, labels = data[p], labels[p]

    train_data_len = int(np.floor(len(data) * train_size_percent))
    return data[0:train_data_len], labels[0:train_data_len], data[train_data_len:], labels[train_data_len:]


def load_data(data_path, file_names, data_type='ML'):
    data = []
    filter = [0, 6]
    for name in file_names:
        d = pd.read_csv(data_path + data_type + '\\' + name, sep=',', header=0,
                        dtype={"Flow Bytes/s": str, " Flow Packets/s": str}, encoding='latin1')
        d["Flow Bytes/s"] = np.float64(d["Flow Bytes/s"])
        d[" Flow Packets/s"] = np.float64(d[" Flow Packets/s"])
        d = d.to_numpy()
        if data_type is 'GL':
            d = d[:, [i for i in range(1, d.shape[1]) if i not in filter]]
            d[:, [0]] = [[int(ipaddress.IPv4Address(ipstr[0]))] for ipstr in d[:, [0]]]
            d[:, [2]] = [[int(ipaddress.IPv4Address(ipstr[0]))] for ipstr in d[:, [2]]]
        d[:, 0:(d.shape[1] - 1)] = np.nan_to_num(d[:, 0:(d.shape[1] - 1)].astype('float64'), copy=True)
        data.append((name, d))
    return data


def encode_labels(labels):
    global LABEL_ENCODER
    LABEL_ENCODER = LabelEncoder()
    LABEL_ENCODER.fit(labels)
    return LABEL_ENCODER.transform(labels)


def get_class_weights(labels, balance=True):
    if balance:
        class_weights = compute_class_weight('balanced', np.unique(labels), labels)
    else:
        class_weights = compute_class_weight(None, np.unique(labels), labels)
    return dict(enumerate(class_weights))


def generate_mlp_model(trainX, trainY, testX, testY, optimizer, learning_rate):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(512, input_shape=(trainX.shape[1],), activation='relu'))
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dropout(rate=0.5))
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dense(trainY.shape[1], activation='sigmoid'))

    try:
        optimizer = getattr(keras.optimizers, optimizer)(lr=learning_rate)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        return model, trainX, trainY, testX, testY
    except AttributeError:
        raise ValueError(f'Optimizer {optimizer} not found!')


def generate_cnn_model(trainX, trainY, testX, testY, optimizer, learning_rate):
    trainX = trainX.reshape((trainX.shape[0], trainX.shape[1], 1))
    testX = testX.reshape((testX.shape[0], testX.shape[1], 1))

    model = keras.models.Sequential()
    model.add(keras.layers.Conv1D(64, 8, padding='same', input_shape=(trainX.shape[1], 1),
                                  data_format='channels_first', activation='relu'))
    model.add(keras.layers.Conv1D(64, 8, padding='same', data_format='channels_first', activation='relu'))
    model.add(keras.layers.MaxPooling1D(padding='same', data_format='channels_first'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(rate=0.5))
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dense(trainY.shape[1], activation='sigmoid'))
    try:
        optimizer = getattr(keras.optimizers, optimizer)(lr=learning_rate)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        return model, trainX, trainY, testX, testY
    except AttributeError:
        raise ValueError(f'Optimizer {optimizer} not found!')


def generate_lstm_model(trainX, trainY, testX, testY, optimizer, learning_rate):
    trainX = trainX.reshape((trainX.shape[0], 1, trainX.shape[1]))
    testX = testX.reshape((testX.shape[0], 1, testX.shape[1]))

    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(units=128, input_shape=(1, trainX.shape[2]), activation='relu'))
    model.add(keras.layers.Dropout(rate=0.5))
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dense(trainY.shape[1], activation='sigmoid'))
    try:
        optimizer = getattr(keras.optimizers, optimizer)(lr=learning_rate)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        return model, trainX, trainY, testX, testY
    except AttributeError:
        raise ValueError(f'Optimizer {optimizer} not found!')


def generate_cnn_lstm_model(trainX, trainY, testX, testY, optimizer, learning_rate):
    trainX = trainX.reshape((trainX.shape[0], trainX.shape[1], 1))
    testX = testX.reshape((testX.shape[0], testX.shape[1], 1))

    model = keras.models.Sequential()
    model.add(keras.layers.Conv1D(64, 8, padding='same', input_shape=(trainX.shape[1], 1),
                                  data_format='channels_first', activation='relu'))
    model.add(keras.layers.LSTM(units=128, activation='relu'))
    model.add(keras.layers.Dropout(rate=0.5))
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dense(trainY.shape[1], activation='sigmoid'))
    try:
        optimizer = getattr(keras.optimizers, optimizer)(lr=learning_rate)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        return model, trainX, trainY, testX, testY
    except AttributeError:
        raise ValueError(f'Optimizer {optimizer} not found!')


def get_model_generator(model_type):
    if model_type is Model.MLP:
        return generate_mlp_model
    if model_type is Model.CNN:
        return generate_cnn_model
    if model_type is Model.LSTM:
        return generate_lstm_model
    if model_type is Model.CNN_LSTM:
        return generate_cnn_lstm_model
    raise ValueError(f'Model type {model_type} not found!')


def duplicate_data(data):
    coefs = {}
    for i in range(len(data)):
        unique, counts = np.unique(data[i][1][:, -1], return_counts=True)
        for j in range(len(unique)):
            if unique[j] in coefs:
                coefs[unique[j]] = coefs[unique[j]] + counts[j]
            else:
                coefs[unique[j]] = counts[j]
        '''maximum = float(np.max(list(coefs.values())))
    for key in coefs:
        coefs[key] = int(np.floor(maximum / float(coefs[key])))'''
    for key in coefs.keys():
        if key is 'BENIGN':
            coefs[key] = 1
        else:
            coefs[key] = 4
    for i in range(len(data)):
        repeats = [coefs[label] for label in data[i][1][:, -1]]
        data[i][1] = np.repeat(data[i][1], repeats, axis=0)
    return data


def train_model(data,
                model_type,
                train_size_percent=0.8,
                shuffle=True,
                normalize=True,
                use_label=False,
                duplicate=True,
                optimizer='Adam',
                learning_rate=0.1,
                epochs=100):
    """
    Funkce k natrenovani a testovani zvoleneho modelu
    :param ndarray data: Data k trenovani a testovani modelu, list tuplu ve tvaru (nazev souboru, data)
    :param Model model_type: Nazev modelu
    :param float train_size_percent: Velikost trenovacich dat
    :param bool shuffle: Michani dat
    :param bool normalize: Normalizace dat
    :param bool use_label: Pouzit i label jako vstup
    :param bool duplicate: Pouzit duplikace misto vyvazovani
    :param str optimizer: Zvoleny optimizer
    :param int epochs: Pocet opakovani trenovani
    :param float learning_rate: Zvoleny learning rate
    """
    if duplicate:
        data = duplicate_data(data)
    if use_label:
        reduce_param = 0
    else:
        reduce_param = 1
    data = np.concatenate(tuple([d[1] for d in data]), axis=0)
    Y = encode_labels(data[:, -1])  # Zakodovani labelu ze stringu na int
    class_weights = get_class_weights(Y, not duplicate)
    data[:, -1] = Y  # Nahrazeni hodnot v puvodnim poli
    if normalize:
        data = preprocessing.normalize(data, axis=0, copy=False)  # Normalizace dat
    Y = keras.utils.to_categorical(Y)  # Prevod labelu na one-hot kodovani
    trainX, trainY, testX, testY = split_data(data, Y, train_size_percent=0.8, shuffle=shuffle)
    trainX = trainX[:, 0:(trainX.shape[1] - reduce_param)]  # Vyber priznaku pro trenovani
    testX = testX[:, 0:(testX.shape[1] - reduce_param)]  # Vyber priznaku pro testovani

    model, trainX, trainY, testX, testY = get_model_generator(model_type)(trainX,
                                                                          trainY,
                                                                          testX,
                                                                          testY,
                                                                          optimizer,
                                                                          learning_rate)
    params = {
        "train_size_percent": train_size_percent,
        "shuffle": shuffle,
        "normalize": normalize,
        "use_label": use_label,
        "duplication": duplicate,
        "optimizer": optimizer,
        "optimizer_learning_rate": learning_rate,
        "loss": model.loss,
        "metrics": model.metrics
    }

    graph_folder = f'.\\Graph\\{model_type.name}\\{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}\\'
    os.makedirs(graph_folder)
    with open(graph_folder + 'model_params.json', mode='w') as f:
        f.write(json.dumps(params, ensure_ascii=False))

    keras.utils.plot_model(model, to_file=graph_folder + 'model.png', show_shapes=True)
    tb_callback = keras.callbacks.TensorBoard(log_dir=graph_folder, histogram_freq=0,
                                              write_graph=True, write_images=True)
    model.fit(trainX, trainY, epochs=epochs, batch_size=10000, validation_data=(testX, testY), shuffle=shuffle,
              callbacks=[tb_callback], class_weight=class_weights)
    model_json = model.to_json()
    with open(graph_folder + 'model.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(graph_folder + 'model.h5')


if __name__ == "__main__":
    import gc
    data = load_data(DATA_FOLDER, [str(i) + '.csv' for i in range(1, 9)], 'GL')
    train_model(data=np.array(data, copy=False),
                model_type=Model.MLP,
                train_size_percent=0.8,
                shuffle=True,
                normalize=False,
                use_label=False,
                duplicate=True,
                optimizer='Adam',
                learning_rate=0.1,
                epochs=10)
    gc.collect()
    train_model(data=np.array(data, copy=False),
                model_type=Model.CNN,
                train_size_percent=0.8,
                shuffle=True,
                normalize=False,
                use_label=False,
                duplicate=True,
                optimizer='Adam',
                learning_rate=0.1,
                epochs=10)
    gc.collect()
    train_model(data=np.array(data, copy=False),
                model_type=Model.LSTM,
                train_size_percent=0.8,
                shuffle=True,
                normalize=False,
                use_label=False,
                duplicate=True,
                optimizer='Adam',
                learning_rate=0.1,
                epochs=10)
    gc.collect()
    train_model(data=np.array(data, copy=False),
                model_type=Model.CNN_LSTM,
                train_size_percent=0.8,
                shuffle=True,
                normalize=False,
                use_label=False,
                duplicate=True,
                optimizer='Adam',
                learning_rate=0.1,
                epochs=10)
    print("Konec")
