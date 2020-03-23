import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import scipy.io
filePattern_train = ["raw/parsed_P0"+str(x)+"T.mat" for x in range(1,9)]
raw_data = []
labels = []

def standardize(data):
    return (data-np.mean(data))/np.std(data)

for train_file in filePattern_train:
    mat = scipy.io.loadmat(train_file)
    #print(mat['RawEEGData'][0][0])
    raw_data.append(standardize(np.array(mat['RawEEGData'])))
    labels.append(mat['Labels']-1)
    # print((raw_data[0][0][0][0:4]))
    # df = pd.DataFrame(raw_data[0][0][0][0:200])
    # df.plot(figsize=(30,5))
    # plt.show()
    #print(df)
np_data = np.concatenate(raw_data,axis=0)
np_labels = np.concatenate(labels,axis=0)
print(np_labels.shape,np_data.shape)



np_data = np.array(np.concatenate(np.split(np_data,16,axis=2),axis=0))


def flat_multiply():
    flat_data = []
    for xi in np_data:
        result = xi[0]
        for row in range(1, len(xi)):
            result *= xi[row]
        flat_data.append(result)
    return flat_data


#np_data=np_data.reshape((-1,256,12))
flat_data =flat_multiply()

np_data= np.array(flat_data)
print(np_labels.shape,np_data.shape)
np_labels=np_labels.repeat(16)[:,np.newaxis]




#b=pd.Panel(rollaxis(np_data,2)).to_frame()
#c=b.set_index(b.index.labels[0]).reset_index()

def splitData(data,label):
    np.random.seed(4)
    np.random.shuffle(data)
    indx = int(0.8 * data.shape[0])
    training_inp, test_inp = data[:indx, :], data[indx:, :]

    np.random.seed(4)
    np.random.shuffle(label)
    training_out, test_out = label[:indx, :], label[indx:, :]
    return training_inp,training_out,test_inp,test_out

training_inp,training_out,test_inp,test_out= splitData(np_data,np_labels)

import pickle as pkl
pkl.dump((training_inp,training_out,test_inp,test_out),open("train_data256.pkl",'wb'))
print("Done")

import os
from pathlib import Path
path = Path(__file__)
path = os.path.join(path.parent,"raw","parsed_P0")
# 2-input XOR inputs and expected outputs.
filePattern_train = [path + str(x) + "T.mat" for x in
                     range(1, 9)]
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
# data_parsed = []
# for xi in range(len(data_inputs)):
#     entry = data_inputs[xi]
#     result = entry[0]
#     for xrow in range(1,len(entry)):
#         result*=entry[xrow]
#     data_parsed.append(result)
# data_inputs = np.array(data_parsed)
data_outputs = np.concatenate(labels, axis=0)
data_inputs=data_inputs.reshape(-1,data_inputs.shape[-1])
data_outputs=data_outputs.repeat(12)[:,np.newaxis]
print(data_outputs.shape, data_inputs.shape)

training_inp,training_out,test_inp,test_out= splitData(data_inputs,data_outputs)
print(training_out.shape,test_out.shape)
pkl.dump((training_inp,training_out,test_inp,test_out),open("train_data4096.pkl",'wb'))