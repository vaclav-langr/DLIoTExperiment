import pandas as pd
import numpy as np
import ipaddress
import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import json
import sklearn.preprocessing as preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from keras.layers import Input, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, LSTM
from keras.models import Model
from keras.utils import to_categorical, plot_model
from keras.callbacks import TensorBoard
from keras import optimizers
from enum import Enum

DATA_FOLDER = 'E:\\Downloads\\Experiment\\'
LABEL_ENCODER = None


class ModelType(Enum):
    MLP = 'MLP'
    CNN = 'CNN'
    LSTM = 'LSTM'
    CNN_LSTM = 'CNN_LSTM'


def split_data(input_data, labels, train_size_percent=0.8, shuffle=False):
    if shuffle:
        p = np.random.permutation(len(input_data))
        input_data, labels = input_data[p], labels[p]
    train_data_len = int(np.floor(len(input_data) * train_size_percent))
    return input_data[0:train_data_len], labels[0:train_data_len], input_data[train_data_len:], labels[train_data_len:]


def check_data(input_data):
    for t in input_data:
        d = t[1]
        for i in range(d.shape[0]):
            sample = np.array(d[i, 0:(d.shape[1] - 1)], dtype=np.float64)
            if np.isnan(sample).any() or np.isinf(sample).any():
                print('Chyba')


def load_data(data_path, file_names, data_type='ML'):
    result = []
    filter_data = [0, 6]
    for name in file_names:
        d = pd.read_csv(data_path + data_type + '\\' + name, sep=',', header=0,
                        dtype={"Flow Bytes/s": str, " Flow Packets/s": str}, encoding='latin1')
        d["Flow Bytes/s"] = np.float64(d["Flow Bytes/s"])
        d[" Flow Packets/s"] = np.float64(d[" Flow Packets/s"])
        d = d.to_numpy()
        if data_type is 'GL':
            d = d[:, [i for i in range(1, d.shape[1]) if i not in filter_data]]
            d[:, [0]] = [[int(ipaddress.IPv4Address(ipstr[0]))] for ipstr in d[:, [0]]]
            d[:, [2]] = [[int(ipaddress.IPv4Address(ipstr[0]))] for ipstr in d[:, [2]]]
        d[:, 0:(d.shape[1] - 1)] = np.array(d[:, 0:(d.shape[1] - 1)], dtype=np.float32)
        for i in range(d.shape[0]):
            d[i, 0:(d.shape[1] - 1)] = np.nan_to_num(np.array(d[i, 0:(d.shape[1] - 1)], dtype=np.float32), copy=True)
        result.append((name, d))
    return result


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
    output_shape = trainY.shape[1] if len(trainY.shape) > 1 else 1

    net_input = Input(shape=(trainX.shape[1],))
    net_output = Dense(512, use_bias=True, activation=None,
                       kernel_initializer='glorot_uniform', bias_initializer='zeros')(net_input)
    net_output = Dense(512, use_bias=True, activation=None,
                       kernel_initializer='glorot_uniform', bias_initializer='zeros')(net_output)
    net_output = Dense(512, use_bias=True, activation=None,
                       kernel_initializer='glorot_uniform', bias_initializer='zeros')(net_output)
    net_output = Dropout(rate=0.5)(net_output)
    net_output = Dense(512, use_bias=True, activation=None,
                       kernel_initializer='glorot_uniform', bias_initializer='zeros')(net_output)
    net_output = Dense(output_shape, activation='sigmoid', use_bias=True,
                       kernel_initializer='glorot_uniform', bias_initializer='zeros')(net_output)
    model = Model(net_input, net_output)

    try:
        optimizer_var = getattr(optimizers, optimizer)(lr=learning_rate)
        model.compile(optimizer=optimizer_var, loss='binary_crossentropy', metrics=['binary_accuracy'])
        return model, trainX, trainY, testX, testY
    except AttributeError:
        raise ValueError(f'Optimizer {optimizer} not found!')


def generate_cnn_model(trainX, trainY, testX, testY, optimizer, learning_rate):
    trainX = trainX.reshape((trainX.shape[0], trainX.shape[1], 1))
    testX = testX.reshape((testX.shape[0], testX.shape[1], 1))

    output_shape = trainY.shape[1] if len(trainY.shape) > 1 else 1

    net_input = Input(shape=(trainX.shape[1], 1))
    net_output = Conv1D(64, 8, padding='same', activation='relu', data_format='channels_first',
                        use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(net_input)
    net_output = Conv1D(64, 8, padding='same', data_format='channels_first', activation='relu',
                        use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(net_output)
    net_output = MaxPooling1D(padding='same', data_format='channels_first')(net_output)
    net_output = Flatten()(net_output)
    net_output = Dropout(rate=0.5)(net_output)
    net_output = Dense(512, use_bias=True, activation=None, kernel_initializer='glorot_uniform',
                       bias_initializer='zeros')(net_output)
    net_output = Dense(output_shape, activation='sigmoid', use_bias=True,
                       kernel_initializer='glorot_uniform', bias_initializer='zeros')(net_output)
    model = Model(net_input, net_output)

    try:
        optimizer = getattr(optimizers, optimizer)(lr=learning_rate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])
        return model, trainX, trainY, testX, testY
    except AttributeError:
        raise ValueError(f'Optimizer {optimizer} not found!')


def generate_lstm_model(trainX, trainY, testX, testY, optimizer, learning_rate):
    trainX = trainX.reshape((trainX.shape[0], 1, trainX.shape[1]))
    testX = testX.reshape((testX.shape[0], 1, testX.shape[1]))

    output_shape = trainY.shape[1] if len(trainY.shape) > 1 else 1

    net_input = Input(shape=(1, trainX.shape[2]))
    net_output = LSTM(units=128, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform',
                      bias_initializer='zeros')(net_input)
    net_output = Dropout(rate=0.5)(net_output)
    net_output = Dense(512, use_bias=True, activation=None, kernel_initializer='glorot_uniform',
                       bias_initializer='zeros')(net_output)
    net_output = Dense(output_shape, activation='sigmoid', use_bias=True,
                       kernel_initializer='glorot_uniform', bias_initializer='zeros')(net_output)
    model = Model(net_input, net_output)

    try:
        optimizer = getattr(optimizers, optimizer)(lr=learning_rate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])
        return model, trainX, trainY, testX, testY
    except AttributeError:
        raise ValueError(f'Optimizer {optimizer} not found!')


def generate_cnn_lstm_model(trainX, trainY, testX, testY, optimizer, learning_rate):
    trainX = trainX.reshape((trainX.shape[0], trainX.shape[1], 1))
    testX = testX.reshape((testX.shape[0], testX.shape[1], 1))

    output_shape = trainY.shape[1] if len(trainY.shape) > 1 else 1

    net_input = Input(shape=(trainX.shape[1], 1))
    net_output = Conv1D(64, 8, padding='same', activation='relu', data_format='channels_first',
                        use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(net_input)
    net_output = LSTM(units=128, use_bias=True, activation='tanh', kernel_initializer='glorot_uniform',
                      bias_initializer='zeros')(net_output)
    net_output = Dropout(rate=0.5)(net_output)
    net_output = Dense(512, use_bias=True, activation=None, kernel_initializer='glorot_uniform',
                       bias_initializer='zeros')(net_output)
    net_output = Dense(output_shape, activation='sigmoid', use_bias=True,
                       kernel_initializer='glorot_uniform', bias_initializer='zeros')(net_output)
    model = Model(net_input, net_output)

    try:
        optimizer = getattr(optimizers, optimizer)(lr=learning_rate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])
        return model, trainX, trainY, testX, testY
    except AttributeError:
        raise ValueError(f'Optimizer {optimizer} not found!')


def get_model_generator(model_type):
    if model_type is ModelType.MLP:
        return generate_mlp_model
    if model_type is ModelType.CNN:
        return generate_cnn_model
    if model_type is ModelType.LSTM:
        return generate_lstm_model
    if model_type is ModelType.CNN_LSTM:
        return generate_cnn_lstm_model
    raise ValueError(f'Model type {model_type} not found!')


def duplicate_data(input_data, duplicate_coef=None):
    coefs = {}
    unique, counts = np.unique(input_data[:, -1], return_counts=True)
    for j in range(len(unique)):
        coefs[unique[j]] = counts[j]
    maximum = float(np.max(list(coefs.values())))
    for key in coefs:
        if duplicate_coef is None:
            coefs[key] = int(np.floor(maximum / float(coefs[key])))
        else:
            coefs[key] = 1 if coefs[key] == maximum else duplicate_coef
    repeats = [coefs[label] for label in input_data[:, -1]]
    input_data = np.repeat(input_data, repeats, axis=0)
    return input_data


def simplify_labels(labels):
    b_indices = np.where(np.array([label == 'BENIGN' for label in labels]))[0]
    a_indices = np.where(np.array([label != 'BENIGN' for label in labels]))[0]
    labels[a_indices] = 'ATTACK'
    labels[b_indices] = 'NORMAL'
    return labels


def normalize_data(input_data):
    std_scaler = preprocessing.StandardScaler().fit(input_data)
    return std_scaler.transform(input_data)


def train_model(input_data,
                model_type,
                train_size_percent=0.8,
                shuffle=True,
                normalize=True,
                use_label=False,
                duplicate=True,
                duplicate_coef=None,
                use_weights=False,
                simplify=True,
                optimizer='Adam',
                learning_rate=0.1,
                epochs=100):
    """
    Funkce k natrenovani a testovani zvoleneho modelu
    :param ndarray input_data: Data k trenovani a testovani modelu, list tuplu ve tvaru (nazev souboru, data)
    :param ModelType model_type: Nazev modelu
    :param float train_size_percent: Velikost trenovacich dat
    :param bool shuffle: Michani dat
    :param bool normalize: Normalizace dat
    :param bool use_label: Pouzit i label jako vstup
    :param bool duplicate: Pouzit duplikace misto vyvazovani
    :param int duplicate_coef: Kolikrat se maji zopakovat nevybalancovane tridy, None - automaticky pocet
    :param bool use_weights: Pouzit vahy pro tridy
    :param bool simplify: Zjednoduseni labelu na 2 tridy, attack a normal
    :param str optimizer: Zvoleny optimizer
    :param int epochs: Pocet opakovani trenovani
    :param float learning_rate: Zvoleny learning rate
    """
    input_data = np.concatenate(tuple([d[1] for d in input_data]), axis=0)
    if simplify:
        input_data[:, -1] = simplify_labels(input_data[:, -1])
    if normalize:
        # Normalizace dat
        input_data[:, 0:(input_data.shape[1] - 1)] = normalize_data(input_data[:, 0:(input_data.shape[1] - 1)])
    if duplicate:
        input_data = duplicate_data(input_data, duplicate_coef)
    if use_label:
        reduce_param = 0
    else:
        reduce_param = 1
    Y = encode_labels(input_data[:, -1])  # Zakodovani labelu ze stringu na int
    class_weights = get_class_weights(Y, use_weights)
    input_data[:, -1] = Y  # Nahrazeni hodnot v puvodnim poli
    #  Y = to_categorical(Y)  # Prevod labelu na one-hot kodovani
    trainX, trainY, testX, testY = split_data(input_data, Y, train_size_percent=train_size_percent, shuffle=shuffle)
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
        "use_weights": use_weights,
        "simplify": simplify,
        "optimizer": optimizer,
        "optimizer_learning_rate": learning_rate,
        "loss": model.loss,
        "metrics": model.metrics
    }

    graph_folder = f'.\\Graph\\{model_type.name}\\{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}\\'
    os.makedirs(graph_folder)
    with open(graph_folder + 'model_params.json', mode='w') as f:
        f.write(json.dumps(params, ensure_ascii=False))

    try:
        plot_model(model, to_file=graph_folder + 'model.png', show_shapes=True)
    except:
        print("GraphViz not found in PATH variable")
    tb_callback = TensorBoard(log_dir=graph_folder, histogram_freq=0,
                              write_graph=True, write_images=True)
    model.fit(trainX, trainY, epochs=epochs, batch_size=1000, validation_data=(testX, testY), shuffle=shuffle,
              callbacks=[tb_callback], class_weight=class_weights, verbose=2)
    model_json = model.to_json()
    with open(graph_folder + 'model.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(graph_folder + 'model.h5')


def svm_model(input_data,
              train_size_percent=0.8,
              shuffle=True,
              normalize=True,
              use_label=False,
              duplicate=True,
              duplicate_coef=None,
              use_weights=False,
              simplify=True):
    input_data = np.concatenate(tuple([d[1] for d in input_data]), axis=0)
    if simplify:
        input_data[:, -1] = simplify_labels(input_data[:, -1])
    if normalize:
        # Normalizace dat
        input_data[:, 0:(input_data.shape[1] - 1)] = normalize_data(input_data[:, 0:(input_data.shape[1] - 1)])
    if duplicate:
        input_data = duplicate_data(input_data, duplicate_coef)
    if use_label:
        reduce_param = 0
    else:
        reduce_param = 1
    Y = encode_labels(input_data[:, -1])  # Zakodovani labelu ze stringu na int
    class_weights = get_class_weights(Y, use_weights)
    input_data[:, -1] = Y  # Nahrazeni hodnot v puvodnim poli
    trainX, trainY, testX, testY = split_data(input_data, Y, train_size_percent=train_size_percent, shuffle=shuffle)
    trainX = trainX[:, 0:(trainX.shape[1] - reduce_param)]  # Vyber priznaku pro trenovani
    testX = testX[:, 0:(testX.shape[1] - reduce_param)]  # Vyber priznaku pro testovani
    clf = svm.SVC(gamma='scale', decision_function_shape='ovo', class_weight=class_weights, verbose=True, max_iter=1000)
    print('Starting training')
    clf.fit(trainX, trainY)
    print('Trained')
    s = clf.score(testX, testY)
    print(f'Test result: {s}')


def forest_model(input_data,
                 train_size_percent=0.8,
                 shuffle=True,
                 normalize=True,
                 use_label=False,
                 duplicate=True,
                 duplicate_coef=None,
                 use_weights=False,
                 simplify=True):
    input_data = np.concatenate(tuple([d[1] for d in input_data]), axis=0)
    if simplify:
        input_data[:, -1] = simplify_labels(input_data[:, -1])
    if normalize:
        # Normalizace dat
        input_data[:, 0:(input_data.shape[1] - 1)] = normalize_data(input_data[:, 0:(input_data.shape[1] - 1)])
    if duplicate:
        input_data = duplicate_data(input_data, duplicate_coef)
    if use_label:
        reduce_param = 0
    else:
        reduce_param = 1
    Y = encode_labels(input_data[:, -1])  # Zakodovani labelu ze stringu na int
    class_weights = get_class_weights(Y, use_weights)
    input_data[:, -1] = Y  # Nahrazeni hodnot v puvodnim poli
    trainX, trainY, testX, testY = split_data(input_data, Y, train_size_percent=train_size_percent, shuffle=shuffle)
    trainX = trainX[:, 0:(trainX.shape[1] - reduce_param)]  # Vyber priznaku pro trenovani
    testX = testX[:, 0:(testX.shape[1] - reduce_param)]  # Vyber priznaku pro testovani
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight=class_weights, verbose=1)
    print('Starting training')
    clf.fit(trainX, trainY)
    print('Trained')
    s = clf.score(testX, testY)
    print(f'Test result: {s}')


if __name__ == "__main__":
    data = load_data(DATA_FOLDER, [str(i) + '.csv' for i in range(1, 9)], 'GL')
    '''forest_model(data=np.array(data, copy=False),
                 train_size_percent=0.8,
                 shuffle=True,
                 normalize=True,
                 use_label=False,
                 duplicate=True,
                 duplicate_coef=None,
                 use_weights=False,
                 simplify=True)'''
    '''svm_model(data=np.array(data, copy=False),
              train_size_percent=0.8,
              shuffle=True,
              normalize=True,
              use_label=False,
              duplicate=True,
              duplicate_coef=None,
              use_weights=False,
              simplify=True)'''
    '''train_model(input_data=np.array(data, copy=False),
                model_type=ModelType.MLP,
                train_size_percent=0.8,
                shuffle=True,
                normalize=True,
                use_label=False,
                duplicate=False,
                duplicate_coef=None,
                use_weights=True,
                simplify=True,
                optimizer='SGD',
                learning_rate=0.1,
                epochs=100)'''
    '''train_model(input_data=np.array(data, copy=False),
                model_type=ModelType.CNN,
                train_size_percent=0.8,
                shuffle=True,
                normalize=True,
                use_label=False,
                duplicate=False,
                duplicate_coef=None,
                use_weights=True,
                simplify=True,
                optimizer='SGD',
                learning_rate=0.1,
                epochs=100)'''
    '''train_model(input_data=np.array(data, copy=False),
                model_type=ModelType.LSTM,
                train_size_percent=0.8,
                shuffle=True,
                normalize=True,
                use_label=False,
                duplicate=False,
                duplicate_coef=None,
                use_weights=True,
                simplify=True,
                optimizer='Adam',
                learning_rate=0.1,
                epochs=100)'''
    train_model(input_data=np.array(data, copy=False),
                model_type=ModelType.CNN_LSTM,
                train_size_percent=0.8,
                shuffle=True,
                normalize=True,
                use_label=False,
                duplicate=False,
                duplicate_coef=None,
                use_weights=True,
                simplify=True,
                optimizer='Adam',
                learning_rate=0.1,
                epochs=50)
    print("Konec")
