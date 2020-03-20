import pickle

import scipy.io
import numpy as np
import ann as ANN

filePattern_train = ["/home/augt/Public/MIN/MetodeDeNatura/GA Versions/data/raw/parsed_P0" + str(x) + "E.mat" for x in
                     range(1, 11)]

raw_data = []
labels = []
visualize = False


def standardize(data):
    return (data - np.mean(data)) / np.std(data)


for train_file in filePattern_train:
    mat = scipy.io.loadmat(train_file)
    print(mat.keys())
    raw_data.append(standardize(np.array(mat['RawEEGData'])))
    labels.append(mat['Labels'] - 1)
    # print(len(raw_data), len(raw_data[0]), len(raw_data[0][0]), len(raw_data[0][0][0]))
    # df = pd.DataFrame(raw_data[0][0][0][0:200])
    sfreq = mat['sampRate']  # Sampling frequency

data_inputs = np.concatenate(raw_data, axis=0)
data_outputs = np.concatenate(labels, axis=0)
print(data_outputs.shape, data_inputs.shape)

f = open("/home/augt/Public/MIN/MetodeDeNatura/GA Versions/manual_ga/weights_5_iterations_10%_mutation.pkl", "rb")
best_weights = pickle.load(f)
f.close()

acc, predictions = ANN.predict_outputs(best_weights, data_inputs, data_outputs, activation="sigmoid")

print(acc, predictions)