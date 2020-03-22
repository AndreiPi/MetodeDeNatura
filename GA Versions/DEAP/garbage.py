import json
import random
import struct
from _codecs import decode

from bitarray import bitarray
from deap import base
from deap import creator
from deap import tools
from individual import individual as indv
import numpy as np
from keras import Sequential
from keras.layers import Dense, Flatten

IND_SIZE = [(4096, 16), (16, 32), (32, 64), (768, 2)]

config_path = "/home/augt/Public/MIN/MetodeDeNatura/GA Versions/config.json"
raw_data = []
labels = []
visualize = False

with open(config_path, "r") as fd:
    config = json.load(fd)


def make_model(config, verbose=False):
    first = True
    model = Sequential()
    for layer in config['ANN']:
        if "Dense" in layer:
            if first:
                model.add(
                    Dense(config['ANN'][layer][0], input_shape=(512, 12, 4096)[1:], activation=config['ANN'][layer][1]))
                first = False
            else:
                model.add(Dense(config['ANN'][layer][0], activation=config['ANN'][layer][1]))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    if verbose:
        model.summary()

    return model


def generate_individual(ind_class, size):
    mat = np.zeros(len(size), dtype=np.ndarray)
    for idx, s in enumerate(size):
        mat[idx] = np.random.rand(s[0], s[1])

    print(mat.shape)
    print(mat.flatten().shape)
    individual = ind_class(mat)

    return individual


model = make_model(config, verbose=True)
for idx, w in enumerate(model.get_weights()):
    if idx % 2 == 0:
        print(w.shape)

creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
creator.create('Individual', indv, fitness=creator.FitnessMin)

tbx = base.Toolbox()
tbx.register('individual', generate_individual, creator.Individual, size=IND_SIZE)
tbx.register('population', tools.initRepeat, list, tbx.individual)

# tbx.register("evaluate", EOQ)
# tbx.register("mate", tools.cxOnePoint)
# tbx.register("mutate", tools.mutFlipBit, indpb=0.01)
# tbx.register("select", tools.selTournament, tournsize=5)


indv1 = tbx.individual()
print(dir(indv1))
print(type(indv1))
print('w: ', indv1.weights)
print('vw: ', indv1.vector_weights)
print('b: ', indv1.nparray_to_binarray(indv1.vector_weights))
print('vw: ', indv1.vector_weights)
print('ck: ', indv1.binarray_to_nparray(indv1.nparray_to_binarray(indv1.vector_weights), indv1.vector_weights.shape))
print([x for x in indv1.binary_vector_weights])

# print(type(indv.unflatten(indv.vector_weights, indv.weights)[1][0]))
# print(list(indv._flatten(indv.weights[0])))
# print(toolbox.population(n=20))

a = np.random.rand(5)
b = np.random.rand(5)

sp = a.shape
# print(a)
# print(b)
# print(tools.cxOnePoint(a, b))

# print('\n\n\n\n')
# m = model.get_weights()
# print(m[0].shape, m[1].shape)
# new_w = []
# for idx, w in enumerate(m):
#     if idx % 2 == 0:
#         print(w.shape, indv1.weights[int(idx / 2)].shape)
#         assert w.shape == indv1.weights[int(idx / 2)].shape
#         new_w.append(indv1.weights[int(idx / 2)])
#     else:
#         new_w.append(w)
#
# model.set_weights(new_w)
# print(model.get_weights())
