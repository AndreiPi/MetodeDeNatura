from __future__ import print_function
import os
import neat
import visualize
import numpy as np
import scipy.io
import os
import pickle
from pathlib import Path
import multiprocessing as mp
from joblib import Parallel, delayed
num_cores = mp.cpu_count()
import time
#print(num_cores)


def eval_genome(genome,config,training_inp,training_out):
    genome.fitness = 3000.0
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    acc = 0
    for xi, xo in zip(training_inp, training_out):
        output = net.activate(xi)
        acc += int(round(output[0]) == xo[0])
        genome.fitness -= (output[0] - xo[0]) ** 2
    return (acc / len(training_out),genome.fitness)


def eval_genomes(genomes, config):
    avg_acc = []

    # with mp.Pool(processes=num_cores-9) as p:
    #     evaluated_genomes = p.starmap(eval_genome, [(genome,config,training_inp,training_out) for genome_id,genome in genomes])

    for genome_id,genome in genomes:
        genome.fitness = 3000.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        acc = 0
        #start = time.time()
        with mp.Pool(processes=num_cores-6) as p:
            processed_list = p.map(net.activate, [xi for xi in training_inp])
        for pi, xo in zip(processed_list, training_out):
            output = np.argmax(pi)
            acc += int(output==xo[0])
            genome.fitness -= (output - xo[0]) ** 2
        avg_acc.append(acc/len(training_out))
    print(np.mean(avg_acc))


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(10))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 50)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for xi, xo in zip(training_inp, training_out):
        output = winner_net.activate(xi)
        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))


    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-19'
                                             '')
    p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    path = Path(__file__)
    path = os.path.join(path.parent.parent, "data", "train_data4096.pkl")
    # 2-input XOR inputs and expected outputs.
    training_inp, training_out, test_inp, test_out = pickle.load(open(path, "rb"))
    training_inp = np.array(training_inp)

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat.config.Config')
    run(config_path)