from keras.models import model_from_json

MODEL_PATH = './Graph/MLP/20190423_065311_naninf/'

with open(MODEL_PATH + 'model.json') as f:
    json_str = " ".join(f.readlines())
    model = model_from_json(json_str)
model.load_weights(MODEL_PATH + 'model.h5')
print(model)