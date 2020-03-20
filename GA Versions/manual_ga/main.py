import json

import numpy
import ga
import pickle
import ann as ANN
import matplotlib.pyplot
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import scipy.io
from matplotlib.collections import LineCollection
import mne

# load config file
config_path = "/home/augt/Public/MIN/MetodeDeNatura/GA Versions/config.json"
with open(config_path, "r") as fd:
    config = json.load(fd)

filePattern_train = ["/home/augt/Public/MIN/MetodeDeNatura/GA Versions/data/raw/parsed_P0" + str(x) + "T.mat" for x in
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
data_outputs = np.concatenate(labels, axis=0)
print(data_outputs.shape, data_inputs.shape)

# f = open("dataset_features.pkl", "rb")
# data_inputs2 = pickle.load(f)
# f.close()
# features_STDs = numpy.std(a=data_inputs2, axis=0)
# data_inputs = data_inputs2[:, features_STDs>50]
#
#
# f = open("outputs.pkl", "rb")
# data_outputs = pickle.load(f)
# f.close()


sol_per_pop = 8
num_parents_mating = 4
num_generations = 5
mutation_percent = 10

# Creating the initial population.
initial_pop_weights = []
for curr_sol in numpy.arange(0, sol_per_pop):
    HL1_neurons = 150
    input_HL1_weights = numpy.random.uniform(low=-0.1, high=0.1, size=(data_inputs.shape[1], HL1_neurons))

    HL2_neurons = 60
    HL1_HL2_weights = numpy.random.uniform(low=-0.1, high=0.1, size=(HL1_neurons, HL2_neurons))

    output_neurons = 4
    HL2_output_weights = numpy.random.uniform(low=-0.1, high=0.1, size=(HL2_neurons, output_neurons))

    initial_pop_weights.append(numpy.array([input_HL1_weights, HL1_HL2_weights, HL2_output_weights]))

pop_weights_mat = numpy.array(initial_pop_weights)
pop_weights_vector = ga.mat_to_vector(pop_weights_mat)

best_outputs = []
accuracies = numpy.empty(shape=(num_generations))

for generation in range(num_generations):
    print("Generation : ", generation)

    # converting the solutions from being vectors to matrices.
    pop_weights_mat = ga.vector_to_mat(pop_weights_vector,
                                       pop_weights_mat)

    # Measuring the fitness of each chromosome in the population.
    fitness = ANN.fitness(pop_weights_mat,
                          data_inputs,
                          data_outputs,
                          activation="sigmoid")

    accuracies[generation] = fitness[0]
    print("Fitness")
    print(fitness)

    # Selecting the best parents in the population for mating.
    parents = ga.select_mating_pool(pop_weights_vector, fitness.copy(), num_parents_mating)
    print("Parents")
    print(parents)

    # Generating next generation using crossover.
    offspring_crossover = ga.crossover(parents, offspring_size=(
    pop_weights_vector.shape[0] - parents.shape[0], pop_weights_vector.shape[1]))

    print("Crossover")
    print(offspring_crossover)

    # Adding some variations to the offsrping using mutation.
    offspring_mutation = ga.mutation(offspring_crossover, mutation_percent=mutation_percent)
    print("Mutation")
    print(offspring_mutation)

    # Creating the new population based on the parents and offspring.
    pop_weights_vector[0:parents.shape[0], :] = parents
    pop_weights_vector[parents.shape[0]:, :] = offspring_mutation

pop_weights_mat = ga.vector_to_mat(pop_weights_vector, pop_weights_mat)
best_weights = pop_weights_mat[0, :]
acc, predictions = ANN.predict_outputs(best_weights, data_inputs, data_outputs, activation="sigmoid")
print("Accuracy of the best solution is : ", acc)

matplotlib.pyplot.plot(accuracies, linewidth=5, color="black")
matplotlib.pyplot.xlabel("Iteration", fontsize=20)
matplotlib.pyplot.ylabel("Fitness", fontsize=20)
matplotlib.pyplot.xticks(numpy.arange(0, num_generations + 1, 100), fontsize=15)
matplotlib.pyplot.yticks(numpy.arange(0, 101, 5), fontsize=15)

f = open("weights_" + str(num_generations) + "_iterations_" + str(mutation_percent) + "%_mutation.pkl", "wb")
pickle.dump(pop_weights_mat, f)
f.close()
