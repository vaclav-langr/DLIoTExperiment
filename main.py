import pandas as pd
import numpy as np
import ipaddress
import datetime
import os
import json
from sklearn.preprocessing import LabelEncoder, normalize
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


def split_data(data, train_size_percent=0.8, shuffle=False):
    train_result = None
    test_result = None
    for d in data:
        train_data_len = int(np.floor(len(d[1]) * train_size_percent))
        if shuffle:
            np.random.shuffle(d[1])

        train_data, test_data = d[1][0:train_data_len], d[1][train_data_len:]
        if train_result is None:
            train_result = train_data
        else:
            train_result = np.concatenate((train_result, train_data), axis=0)

        if test_result is None:
            test_result = test_data
        else:
            test_result = np.concatenate((test_result, test_data), axis=0)
    return train_result, test_result


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


def encode_labels(train, test):
    global LABEL_ENCODER
    LABEL_ENCODER = LabelEncoder()
    LABEL_ENCODER.fit(np.concatenate((train, test)))
    return LABEL_ENCODER.transform(train), LABEL_ENCODER.transform(test)


def binarize_labels(trainY, testY):
    transformed = keras.utils.to_categorical(np.concatenate((trainY, testY)))
    return transformed[0:len(trainY)], transformed[len(trainY):]


def normalize_data(train, test):
    whole_data = np.concatenate((train, test), axis=0)
    normalized_data = normalize(whole_data, axis=0, copy=True)
    return normalized_data[0:len(train)], normalized_data[len(train):]


def get_class_weights(trainY, testY):
    y_integers = np.argmax(np.concatenate((trainY, testY), axis=0), axis=1)
    class_weights = compute_class_weight(None, np.unique(y_integers), y_integers)
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
        return model
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
        return model
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
        return model
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
        return model
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


def train_model(data,
                model_type,
                train_size_percent=0.8,
                shuffle=True,
                normalize=True,
                use_label=False,
                optimizer='Adam',
                learning_rate=0.1):
    """
    Funkce k natrenovani a testovani zvoleneho modelu
    :param dict data: Data k trenovani a testovani modelu
    :param Model model_type: Nazev modelu
    :param float train_size_percent: Velikost trenovacich dat
    :param bool shuffle: Michani dat
    :param bool normalize: Normalizace dat
    :param bool use_label: Pouzit i label jako vstup
    :param str optimizer: Zvoleny optimizer
    :param float learning_rate: Zvoleny learning rate
    """
    if use_label:
        reduce_param = 0
    else:
        reduce_param = 1
    train_data, test_data = split_data(data, train_size_percent, shuffle)  # Rozdeleni dat na trenovaci a testovaci
    if shuffle:
        np.random.shuffle(train_data)
        np.random.shuffle(test_data)
    trainY, testY = encode_labels(train_data[:, -1], test_data[:, -1])  # Zakodovani labelu ze stringu na int
    train_data[:, -1], test_data[:, -1] = trainY, testY  # Nahrazeni hodnot v puvodnim poli
    trainY, testY = binarize_labels(trainY, testY)  # Prevod labelu na one-hot kodovani
    if normalize:
        train_data, test_data = normalize_data(train_data, test_data)  # Normalizace dat
    trainX = train_data[:, 0:(train_data.shape[1] - reduce_param)]  # Vyber priznaku pro trenovani
    testX = test_data[:, 0:(test_data.shape[1] - reduce_param)]  # Vyber priznaku pro testovani

    class_weights = get_class_weights(trainY, testY)

    model = get_model_generator(model_type)(trainX, trainY, testX, testY, optimizer, learning_rate)
    params = {
        "train_size_percent": train_size_percent,
        "shuffle": shuffle,
        "normalize": normalize,
        "use_label": use_label,
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
    model.fit(trainX, trainY, epochs=100, batch_size=10000, validation_data=(testX, testY), shuffle=shuffle,
              callbacks=[tb_callback], class_weight=class_weights)
    model_json = model.to_json()
    with open(graph_folder + 'model.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(graph_folder + 'model.h5')


def mlp_model(data, train_size_percent=0.8, shuffle=True, normalize=True, use_label=False):
    if use_label:
        reduce_param = 0
    else:
        reduce_param = 1
    train_data, test_data = split_data(data, train_size_percent, shuffle)  # Rozdeleni dat na trenovaci a testovaci
    if shuffle:
        np.random.shuffle(train_data)
        np.random.shuffle(test_data)
    trainY, testY = encode_labels(train_data[:, -1], test_data[:, -1])  # Zakodovani labelu ze stringu na int
    train_data[:, -1], test_data[:, -1] = trainY, testY  # Nahrazeni hodnot v puvodnim poli
    trainY, testY = binarize_labels(trainY, testY)  # Prevod labelu na one-hot kodovani
    if normalize:
        train_data, test_data = normalize_data(train_data, test_data)  # Normalizace dat
    trainX = train_data[:, 0:(train_data.shape[1] - reduce_param)]  # Vyber priznaku pro trenovani
    testX = test_data[:, 0:(test_data.shape[1] - reduce_param)]  # Vyber priznaku pro testovani

    class_weights = get_class_weights(trainY, testY)

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(512, input_shape=(trainX.shape[1], ), activation='relu'))
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dropout(rate=0.5))
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dense(trainY.shape[1], activation='sigmoid'))

    optimizer = keras.optimizers.Adam(lr=0.1)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    params = {
        "train_size_percent": train_size_percent,
        "shuffle": shuffle,
        "normalize": normalize,
        "use_label": use_label,
        "optimizer": model.optimizer.__class__.__name__,
        "loss": model.loss,
        "metrics": model.metrics
    }

    graph_folder = './Graph/MLP/' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(graph_folder)
    with open(graph_folder + '\\model_params.json', mode='w') as f:
        f.write(json.dumps(params, ensure_ascii=False))

    keras.utils.plot_model(model, to_file=graph_folder + '\\model.png', show_shapes=True)
    tb_callback = keras.callbacks.TensorBoard(log_dir=graph_folder, histogram_freq=0,
                                              write_graph=True, write_images=True)
    model.fit(trainX, trainY, epochs=100, batch_size=10000, validation_data=(testX, testY), shuffle=shuffle,
              callbacks=[tb_callback], class_weight=class_weights)
    model_json = model.to_json()
    with open(graph_folder + '\\model.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(graph_folder + '\\model.h5')


def cnn_model(data, train_size_percent=0.8, shuffle=False, normalize=True, use_label=False):
    if use_label:
        reduce_param = 0
    else:
        reduce_param = 1
    train_data, test_data = split_data(data, train_size_percent, shuffle)  # Rozdeleni dat na trenovaci a testovaci
    if shuffle:
        np.random.shuffle(train_data)
        np.random.shuffle(test_data)
    trainY, testY = encode_labels(train_data[:, -1], test_data[:, -1])  # Zakodovani labelu ze stringu na int
    train_data[:, -1], test_data[:, -1] = trainY, testY  # Nahrazeni hodnot v puvodnim poli
    trainY, testY = binarize_labels(trainY, testY)  # Prevod labelu na one-hot kodovani
    if normalize:
        train_data, test_data = normalize_data(train_data, test_data)  # Normalizace dat
    trainX = train_data[:, 0:(train_data.shape[1] - reduce_param)]  # Vyber priznaku pro trenovani
    testX = test_data[:, 0:(test_data.shape[1] - reduce_param)]  # Vyber priznaku pro testovani

    trainX = trainX.reshape((trainX.shape[0], trainX.shape[1], 1))
    testX = testX.reshape((testX.shape[0], testX.shape[1], 1))

    class_weights = get_class_weights(trainY, testY)

    model = keras.models.Sequential()

    model.add(keras.layers.Conv1D(64, 8, padding='same', input_shape=(trainX.shape[1], 1),
                                  data_format='channels_first', activation='relu'))
    model.add(keras.layers.Conv1D(64, 8, padding='same', data_format='channels_first', activation='relu'))
    model.add(keras.layers.MaxPooling1D(padding='same', data_format='channels_first'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(rate=0.5))
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dense(trainY.shape[1], activation='sigmoid'))

    optimizer = keras.optimizers.Adam(lr=0.1)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    params = {
        "train_size_percent": train_size_percent,
        "shuffle": shuffle,
        "normalize": normalize,
        "use_label": use_label,
        "optimizer": model.optimizer.__class__.__name__,
        "loss": model.loss,
        "metrics": model.metrics
    }

    graph_folder = './Graph/CNN/' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(graph_folder)
    with open(graph_folder + '\\model_params.json', mode='w') as f:
        f.write(json.dumps(params, ensure_ascii=False))

    keras.utils.plot_model(model, to_file=graph_folder + '\\model.png', show_shapes=True)
    tb_callback = keras.callbacks.TensorBoard(log_dir=graph_folder, histogram_freq=0,
                                              write_graph=True, write_images=True)
    model.fit(trainX, trainY, epochs=100, batch_size=10000, validation_data=(testX, testY), callbacks=[tb_callback],
              shuffle=shuffle, class_weight=class_weights)
    model_json = model.to_json()
    with open(graph_folder + '\\model.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(graph_folder + '\\model.h5')


def lstm_model(data, train_size_percent=0.8, shuffle=False, normalize=True, use_label=False):
    if use_label:
        reduce_param = 0
    else:
        reduce_param = 1
    train_data, test_data = split_data(data, train_size_percent, shuffle)  # Rozdeleni dat na trenovaci a testovaci
    if shuffle:
        np.random.shuffle(train_data)
        np.random.shuffle(test_data)
    trainY, testY = encode_labels(train_data[:, -1], test_data[:, -1])  # Zakodovani labelu ze stringu na int
    train_data[:, -1], test_data[:, -1] = trainY, testY  # Nahrazeni hodnot v puvodnim poli
    trainY, testY = binarize_labels(trainY, testY)  # Prevod labelu na one-hot kodovani
    if normalize:
        train_data, test_data = normalize_data(train_data, test_data)  # Normalizace dat
    trainX = train_data[:, 0:(train_data.shape[1] - reduce_param)]  # Vyber priznaku pro trenovani
    testX = test_data[:, 0:(test_data.shape[1] - reduce_param)]  # Vyber priznaku pro testovani

    trainX = trainX.reshape((trainX.shape[0], 1, trainX.shape[1]))
    testX = testX.reshape((testX.shape[0], 1, testX.shape[1]))

    class_weights = get_class_weights(trainY, testY)

    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(units=128, input_shape=(1, trainX.shape[2]), activation='relu'))
    model.add(keras.layers.Dropout(rate=0.5))
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dense(trainY.shape[1], activation='sigmoid'))

    optimizer = keras.optimizers.Adam(lr=0.1)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    params = {
        "train_size_percent": train_size_percent,
        "shuffle": shuffle,
        "normalize": normalize,
        "use_label": use_label,
        "optimizer": model.optimizer.__class__.__name__,
        "loss": model.loss,
        "metrics": model.metrics
    }

    graph_folder = './Graph/LSTM/' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(graph_folder)
    with open(graph_folder + '\\model_params.json', mode='w') as f:
        f.write(json.dumps(params, ensure_ascii=False))

    keras.utils.plot_model(model, to_file=graph_folder + '\\model.png', show_shapes=True)
    tb_callback = keras.callbacks.TensorBoard(log_dir=graph_folder, histogram_freq=0,
                                              write_graph=True, write_images=True)
    model.fit(trainX, trainY, epochs=100, batch_size=10000, validation_data=(testX, testY), callbacks=[tb_callback],
              shuffle=shuffle, class_weight=class_weights)
    model_json = model.to_json()
    with open(graph_folder + '\\model.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(graph_folder + '\\model.h5')


def cnn_lstm_model(data, train_size_percent=0.8, shuffle=False, normalize=True, use_label=False):
    if use_label:
        reduce_param = 0
    else:
        reduce_param = 1
    train_data, test_data = split_data(data, train_size_percent, shuffle)  # Rozdeleni dat na trenovaci a testovaci
    if shuffle:
        np.random.shuffle(train_data)
        np.random.shuffle(test_data)
    trainY, testY = encode_labels(train_data[:, -1], test_data[:, -1])  # Zakodovani labelu ze stringu na int
    train_data[:, -1], test_data[:, -1] = trainY, testY  # Nahrazeni hodnot v puvodnim poli
    trainY, testY = binarize_labels(trainY, testY)  # Prevod labelu na one-hot kodovani
    if normalize:
        train_data, test_data = normalize_data(train_data, test_data)  # Normalizace dat
    trainX = train_data[:, 0:(train_data.shape[1] - reduce_param)]  # Vyber priznaku pro trenovani
    testX = test_data[:, 0:(test_data.shape[1] - reduce_param)]  # Vyber priznaku pro testovani

    trainX = trainX.reshape((trainX.shape[0], trainX.shape[1], 1))
    testX = testX.reshape((testX.shape[0], testX.shape[1], 1))

    class_weights = get_class_weights(trainY, testY)

    model = keras.models.Sequential()
    model.add(keras.layers.Conv1D(64, 8, padding='same', input_shape=(trainX.shape[1], 1),
                                  data_format='channels_first', activation='relu'))
    model.add(keras.layers.LSTM(units=128, activation='relu'))
    model.add(keras.layers.Dropout(rate=0.5))
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dense(trainY.shape[1], activation='sigmoid'))

    optimizer = keras.optimizers.Adam(lr=0.1)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    params = {
        "train_size_percent": train_size_percent,
        "shuffle": shuffle,
        "normalize": normalize,
        "use_label": use_label,
        "optimizer": model.optimizer.__class__.__name__,
        "loss": model.loss,
        "metrics": model.metrics
    }

    graph_folder = './Graph/CNN_LSTM/' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(graph_folder)
    with open(graph_folder + '\\model_params.json', mode='w') as f:
        f.write(json.dumps(params, ensure_ascii=False))

    keras.utils.plot_model(model, to_file=graph_folder + '\\model.png', show_shapes=True)
    tb_callback = keras.callbacks.TensorBoard(log_dir=graph_folder, histogram_freq=0,
                                              write_graph=True, write_images=True)
    model.fit(trainX, trainY, epochs=100, batch_size=10000, validation_data=(testX, testY), callbacks=[tb_callback],
              shuffle=shuffle, class_weight=class_weights)
    model_json = model.to_json()
    with open(graph_folder + '\\model.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(graph_folder + '\\model.h5')


if __name__ == "__main__":
    data = load_data(DATA_FOLDER, [str(i) + '.csv' for i in range(1, 9)], 'GL')
    mlp_model(np.array(data, copy=True), 0.8, True, True, False)
    cnn_model(np.array(data, copy=True), 0.8, True, True, False)
    lstm_model(np.array(data, copy=True), 0.8, True, True, False)
    cnn_lstm_model(np.array(data, copy=True), 0.8, True, True, False)
    print("Konec")
