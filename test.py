from keras.models import model_from_json
import ipaddress
import numpy as np
import pandas as pd
import keras as K

DATA_FOLDER = 'E:\\Downloads\\Experiment\\'


def load_model(model_path):
    with open(model_path + 'model.json') as f:
        json_str = " ".join(f.readlines())
        _model = model_from_json(json_str)
    _model.load_weights(model_path + 'model.h5')
    return _model


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
        data.append(d)
    return np.concatenate(data, axis=0)


def modify_data(data):
    data[0] = 100


if __name__ == '__main__':
    print()
    #model = load_model('./Graph/MLP/20190423_065311_naninf/')
    #data = load_data(DATA_FOLDER, [str(i) + '.csv' for i in range(1, 9)], 'GL')
    #unique, counts = np.unique(data[:, -1], return_counts=True)
    #print(np.asarray((unique, counts)).T)
