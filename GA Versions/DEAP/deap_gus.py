import array
import json
import random

import scipy.io
import numpy as np
import keras
from keras import Sequential, Input
from keras.layers import Dense, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from individual import individual as indv

def standardize(data):
    return (data - np.mean(data)) / np.std(data)


def make_model(config, verbose=False):
    first = True
    model = Sequential()
    for layer in config['ANN']:
        if "Dense" in layer:
            if first:
                model.add(
                    Dense(config['ANN'][layer][0], input_shape=X_train.shape[1:], activation=config['ANN'][layer][1]))
                first = False
            else:
                model.add(Dense(config['ANN'][layer][0], activation=config['ANN'][layer][1]))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    if verbose:
        model.summary()

    return model


def load_train_test():
    filePattern_train = ["/home/augt/Public/MIN/MetodeDeNatura/GA Versions/data/raw/parsed_P0" + str(x) + "T.mat" for x
                         in
                         range(1, 9)]

    for train_file in filePattern_train:
        mat = scipy.io.loadmat(train_file)
        raw_data.append(standardize(np.array(mat['RawEEGData'])))
        labels.append(mat['Labels'] - 1)
        # print(len(raw_data), len(raw_data[0]), len(raw_data[0][0]), len(raw_data[0][0][0]))
        # df = pd.DataFrame(raw_data[0][0][0][0:200])
        sfreq = mat['sampRate']  # Sampling frequency

    x = np.concatenate(raw_data, axis=0)
    y = np.concatenate(labels, axis=0)
    ohe = OneHotEncoder()
    y = ohe.fit_transform(y).toarray()
    x_train, x_test, _y_train, _y_test = train_test_split(x, y, test_size=0.20, random_state=42)
    return x_train, x_test, _y_train, _y_test


config_path = "/home/augt/Public/MIN/MetodeDeNatura/GA Versions/config.json"
raw_data = []
labels = []
visualize = False

with open(config_path, "r") as fd:
    config = json.load(fd)

X_train, X_test, y_train, y_test = load_train_test()

model = make_model(config)
print(X_train.shape, y_train.shape)

IND_SIZE = []
for idx, w in enumerate(model.get_weights()):
    if idx % 2 == 0:
        IND_SIZE.append(w.shape)


def fittnes_gus(individual):
    # fit
    model = make_model(config)
    new_w = []

    for idx, w in enumerate(model.get_weights()):
        if idx % 2 == 0:
            assert w.shape == individual.weights[int(idx / 2)].shape
            new_w.append(individual.weights[int(idx / 2)])
        else:
            new_w.append(w)

    model.set_weights(new_w)

    # predict
    y_pred = model.predict(X_test)
    # Converting predictions to label
    pred = list()
    for i in range(len(y_pred)):
        pred.append(np.argmax(y_pred[i]))
    # Converting one hot encoded test label to label
    test = list()
    for i in range(len(y_test)):
        test.append(np.argmax(y_test[i]))

    from sklearn.metrics import accuracy_score
    a = accuracy_score(pred, test)
    # print('Accuracy is:', a * 100)
    return a,

def generate_individual(ind_class, size):
    mat = np.zeros(len(size), dtype=np.ndarray)
    for idx, s in enumerate(size):
        mat[idx] = np.random.rand(s[0], s[1])

    individual = ind_class(mat)

    return individual

def crossover_gus(indv1, indv2):

    genes1 = indv1.vector_weights
    genes2 = indv2.vector_weights
    cross1, cross2 = tools.cxOnePoint(genes1, genes2)

    indv1.vector_weights = cross1
    indv2.vector_weights = cross2

    indv1.set_mat_from_vect()
    indv2.set_mat_from_vect()


    return indv1, indv2

def mutate_gus(individual, indpb):
    bit_array = individual.binary_vector_weights
    new_bit_array = tools.mutFlipBit(bit_array, indpb)
    new_bit_array = new_bit_array[0]
    individual.binary_vector_weights = new_bit_array
    individual.vector_weights = individual.binarray_to_nparray(new_bit_array, individual.vector_weights.shape)
    individual.set_mat_from_vect()

    return individual,

creator.create('FitnessMax', base.Fitness, weights=(1.0,))
creator.create('Individual', indv, fitness=creator.FitnessMax)

tbx = base.Toolbox()
tbx.register('individual', generate_individual, creator.Individual, size=IND_SIZE)
tbx.register('population', tools.initRepeat, list, tbx.individual)

tbx.register("evaluate", fittnes_gus)
tbx.register("mate", crossover_gus)
tbx.register("mutate", mutate_gus, indpb=0.01)
tbx.register("select", tools.selTournament, tournsize=5)

random.seed(64)
pop = tbx.population(n=20)


hof = tools.HallOfFame(1, similar=np.array_equal)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

algorithms.eaSimple(pop, tbx, cxpb=0.5, mutpb=0.2, ngen=600, stats=stats,
                    halloffame=hof)

print(pop, stats, hof)