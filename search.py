import mlrose_hiive as mlr
import os
import time
from statistics import mean, stdev
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split,  learning_curve
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn import neural_network


SMALL_SIZE = 10
KNAPSACK_SMALL_WEIGHTS  = [13, 5, 2, 8, 8, 20, 19, 11, 8, 15]
KNAPSACK_SMALL_VALUES   = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
MEDIUM_SIZE = 30
KNAPSACK_MEDIUM_WEIGHTS = [13, 5, 2, 8, 8, 20, 19, 11, 8, 15,
                           19, 15, 16, 9, 4, 7, 12, 17, 6, 2,
                           5, 6, 5, 10, 13, 13, 7, 16, 4, 18]
KNAPSACK_MEDIUM_VALUES  = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                           11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                           21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
LARGE_SIZE = 50
KNAPSACK_LARGE_WEIGHTS  = [13, 5, 2, 8, 8, 20, 19, 11, 8, 15,
                           19, 15, 16, 9, 4, 7, 12, 17, 6, 2,
                           5, 6, 5, 10, 13, 13, 7, 16, 4, 18,
                           18, 1, 3, 19, 17, 18, 11, 20, 14, 7,
                           15, 13, 4, 2, 14, 9, 3, 12, 18, 3]
KNAPSACK_LARGE_VALUES   = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                           11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                           21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                           31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                           41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
NUM_TRIALS = 5


DATABASE = {
    'algorithms': [
        ['Randomized Hill Climbing', mlr.random_hill_climb, 'red'],
        ['Simulated Annealing', mlr.simulated_annealing, 'green'],
        ['Genetic Algorithm', mlr.genetic_alg, 'blue'],
        ['MIMIC', mlr.mimic, 'black']
    ],
    'fitness_functions': {
        'Knapsack': [
            {
                "problem_args": {'length': SMALL_SIZE, 'fitness_fn': mlr.Knapsack(KNAPSACK_SMALL_WEIGHTS, KNAPSACK_SMALL_VALUES, 0.3), 'maximize': True},
                "algorithms_args": [
                    {'max_attempts': 9, 'restarts': 95, 'curve': True},
                    {'schedule': mlr.GeomDecay(init_temp=1.0, decay=0.99, min_temp=0.001), 'max_attempts': 15, 'max_iters': 30, 'curve': True},
                    {'pop_size': 200, 'mutation_prob': 0.001, 'max_attempts': 3, 'max_iters': 4, 'curve': True},
                    {'pop_size': 750, 'keep_pct': 0.01, 'max_attempts': 1, 'max_iters': 1, 'curve': True}
                ]
            },
            {
                "problem_args": {'length': MEDIUM_SIZE, 'fitness_fn': mlr.Knapsack(KNAPSACK_MEDIUM_WEIGHTS, KNAPSACK_MEDIUM_VALUES, 0.3), 'maximize': True},
                "algorithms_args": [
                    {'max_attempts': 1000, 'restarts': 90, 'curve': True},
                    {'schedule': mlr.GeomDecay(init_temp=1.0, decay=0.99, min_temp=0.001), 'max_attempts': 40, 'max_iters': 580, 'curve': True},
                    {'pop_size': 900, 'mutation_prob': 0.005, 'max_attempts': 4, 'max_iters': 18, 'curve': True},
                    {'pop_size': 4000, 'keep_pct': 0.01, 'max_attempts': 1, 'max_iters': 2, 'curve': True}
                ]
            },
            {
                "problem_args": {'length': LARGE_SIZE, 'fitness_fn': mlr.Knapsack(KNAPSACK_LARGE_WEIGHTS, KNAPSACK_LARGE_VALUES, 0.3), 'maximize': True},
                "algorithms_args": [
                    {'max_attempts': 500, 'restarts': 500, 'curve': True},
                    {'schedule': mlr.GeomDecay(init_temp=300.0, decay=0.99, min_temp=15), 'max_attempts': 60, 'max_iters': 1130, 'curve': True},
                    {'pop_size': 950, 'mutation_prob': 0.02, 'max_attempts': 4, 'max_iters': 35, 'curve': True},
                    {'pop_size': 2500, 'keep_pct': 0.001, 'max_attempts': 1, 'max_iters': 3, 'curve': True}
                ]
            }
        ],
        "Flip Flop": [
            {
                "problem_args": {'length': SMALL_SIZE, 'fitness_fn': mlr.FlipFlop(), 'maximize': True},
                "algorithms_args": [
                    {'max_attempts': 10, 'restarts': 20, 'curve': True},
                    {'schedule': mlr.GeomDecay(init_temp=1.0, decay=0.99, min_temp=0.001), 'max_attempts': 14, 'max_iters': 230, 'curve': True},
                    {'pop_size': 600, 'mutation_prob': 0.0, 'max_attempts': 1, 'max_iters': 1, 'curve': True},
                    {'pop_size': 200, 'keep_pct': 0.5, 'max_attempts': 1, 'max_iters': 2, 'curve': True}
                ]
            },
            {
                "problem_args": {'length': MEDIUM_SIZE, 'fitness_fn': mlr.FlipFlop(), 'maximize': True},
                "algorithms_args": [
                    {'max_attempts': 500, 'restarts': 300, 'curve': True},
                    {'schedule': mlr.GeomDecay(init_temp=1.0, decay=0.99, min_temp=0.001), 'max_attempts': 90, 'max_iters': 8000, 'curve': True},
                    {'pop_size': 2000, 'mutation_prob': 0.0, 'max_attempts': 7, 'max_iters': 20, 'curve': True},
                    {'pop_size': 1700, 'keep_pct': 0.2, 'max_attempts': 1, 'max_iters': 3, 'curve': True}
                ]
            },
            {
                "problem_args": {'length': LARGE_SIZE, 'fitness_fn': mlr.FlipFlop(), 'maximize': True},
                "algorithms_args": [
                    {'max_attempts': 300, 'restarts': 500, 'curve': True},
                    {'schedule': mlr.GeomDecay(init_temp=1.0, decay=0.99, min_temp=0.001), 'max_attempts': 200, 'max_iters': 70000, 'curve': True},
                    {'pop_size': 10000, 'mutation_prob': 0.0, 'max_attempts': 10, 'max_iters': 10, 'curve': True},
                    {'pop_size': 10000, 'keep_pct': 0.01, 'max_attempts': 1, 'max_iters': 3, 'curve': True}
                ]
            }
        ],
        'Four Peaks': [
            {
                "problem_args": {'length': SMALL_SIZE, 'fitness_fn': mlr.FourPeaks(t_pct=0.15), 'maximize': True},
                "algorithms_args": [
                    {'max_attempts': 5, 'restarts': 25, 'curve': True},
                    {'schedule': mlr.GeomDecay(init_temp=2.6, decay=0.99, min_temp=1.5), 'max_attempts': 10, 'max_iters': 75, 'curve': True},
                    {'pop_size': 90, 'mutation_prob': 0.0, 'max_attempts': 1, 'max_iters': 3, 'curve': True},
                    {'pop_size': 500, 'keep_pct': 0.009, 'max_attempts': 1, 'max_iters': 1, 'curve': True}
                ]
            },
            {
                "problem_args": {'length': MEDIUM_SIZE, 'fitness_fn': mlr.FourPeaks(t_pct=0.15), 'maximize': True},
                "algorithms_args": [
                    {'max_attempts': 350, 'restarts': 18, 'curve': True},
                    {'schedule': mlr.GeomDecay(init_temp=4.8, decay=0.99, min_temp=0.1), 'max_attempts': 45, 'max_iters': 500, 'curve': True},
                    {'pop_size': 2300, 'mutation_prob': 0.001, 'max_attempts': 2, 'max_iters': 11, 'curve': True},
                    {'pop_size': 20000, 'keep_pct': 0.009, 'max_attempts': 3, 'max_iters': 6, 'curve': True}
                ]
            },
            {
                "problem_args": {'length': LARGE_SIZE, 'fitness_fn': mlr.FourPeaks(t_pct=0.15), 'maximize': True},
                "algorithms_args": [
                    {'max_attempts': 100, 'restarts': 30, 'curve': True},
                    {'schedule': mlr.GeomDecay(init_temp=6.0, decay=0.8, min_temp=0.20), 'max_attempts': 2000, 'max_iters': 1850, 'curve': True},
                    {'pop_size': 7400, 'mutation_prob': 0.001, 'max_attempts': 2, 'max_iters': 10, 'curve': True},
                    {'pop_size': 2000, 'keep_pct': 0.01, 'max_attempts': 1, 'max_iters': 8, 'curve': True}
                ]
            }
        ],
    }
}


def run_search():
    try:
        os.mkdir('plots')
    except FileExistsError:
        pass

    algorithms = DATABASE.get('algorithms')
    for fitness_function_label, sized_runs in DATABASE.get('fitness_functions').items():
        fitnesses = [[] for _ in range(len(algorithms))]
        stdevs = [[] for _ in range(len(algorithms))]
        times = [[] for _ in range(len(algorithms))]
        iters = [[] for _ in range(len(algorithms))]
        for sized_run in sized_runs:
            problem = mlr.DiscreteOpt(**sized_run.get('problem_args'))
            for i in range(len(algorithms)):
                trial_fitnesses = []
                trial_times = []
                trial_iters = []
                for t in range(NUM_TRIALS):
                    algorithm = algorithms[i][1]
                    algorithm_args = sized_run.get('algorithms_args')[i]
                    algorithm_args.update({'problem': problem})
                    algorithm_args.update({'random_state': t + 1})
                    start = time.time()
                    _, best_fitness, curve = algorithm(**algorithm_args)
                    trial_times.append(time.time() - start)
                    trial_fitnesses.append(best_fitness)
                    trial_iters.append(len(curve))

                fitnesses[i].append(mean(trial_fitnesses))
                stdevs[i].append(stdev(trial_fitnesses))
                times[i].append(mean(trial_times))
                iters[i].append(mean(trial_iters))

        plt.title(f'{fitness_function_label} - Fitness')
        plt.xlabel('Problem size')
        plt.ylabel('Average fitness')
        for i in range(len(algorithms)):
            algorithm = algorithms[i]
            plt.plot(['Small', 'Medium', 'Large'], fitnesses[i], color=algorithm[2], label=algorithm[0])
        plt.legend()
        plt.savefig(f'plots/{fitness_function_label.lower().replace(' ', '_')}_fitness.png')
        plt.clf()

        plt.title(f'{fitness_function_label} - Fitness Standard Deviation')
        plt.xlabel('Problem size')
        plt.ylabel('stdev')
        for i in range(len(algorithms)):
            algorithm = algorithms[i]
            plt.plot(['Small', 'Medium', 'Large'], stdevs[i], color=algorithm[2], label=algorithm[0])
        plt.legend()
        plt.savefig(f'plots/{fitness_function_label.lower().replace(' ', '_')}_stdev.png')
        plt.clf()

        plt.title(f'{fitness_function_label} - Wall Clock Time')
        plt.xlabel('Problem size')
        plt.ylabel('Average wall clock time (seconds)')
        for i in range(len(algorithms)):
            algorithm = algorithms[i]
            plt.plot(['Small', 'Medium', 'Large'], times[i], color=algorithm[2], label=algorithm[0])
        plt.legend()
        plt.savefig(f'plots/{fitness_function_label.lower().replace(' ', '_')}_time.png')
        plt.clf()

        plt.title(f'{fitness_function_label} - Iterations')
        plt.xlabel('Problem size')
        plt.ylabel('Num iterations')
        for i in range(len(algorithms)):
            algorithm = algorithms[i]
            plt.plot(['Small', 'Medium', 'Large'], iters[i], color=algorithm[2], label=algorithm[0])
        plt.legend()
        plt.savefig(f'plots/{fitness_function_label.lower().replace(' ', '_')}_iters.png')
        plt.clf()

        plt.title(f'{fitness_function_label} - Fitness v. Iterations')
        plt.xlabel('Fitness')
        plt.ylabel('Iterations')
        for i in range(len(algorithms)):
            algorithm = algorithms[i]
            plt.plot(fitnesses[i], iters[i], color=algorithm[2], label=algorithm[0])
        plt.legend()
        plt.savefig(f'plots/{fitness_function_label.lower().replace(' ', '_')}_fitness_iters.png')
        plt.clf()


SPLIT = 0.80


def train_and_run(X, Y, x, y, net):
    n = net.fit(np.asarray(X), np.asarray(Y))

    Y_pred = net.predict(X)
    Y_acc = accuracy_score(np.asarray(Y), np.asarray(Y_pred))

    y_pred = net.predict(x)
    y_acc = accuracy_score(np.asarray(y), np.asarray(y_pred))

    nc = None
    try:
        nc = n.fitness_curve
    except AttributeError:
        pass

    return 1 - Y_acc, 1 - y_acc, n.loss, nc


def run_neural():
    np.random.seed(1)

    data = get_data('data/credit.csv')
    data_size = len(data)
    y1 = []
    y2 = []
    t = []
    l = []

    split_index = int(SPLIT * data_size)
    scaler = MinMaxScaler()
    one_hot = OneHotEncoder()
    X = scaler.fit_transform(data[: split_index,: -1])
    Y = one_hot.fit_transform(data[: split_index, -1].reshape(-1, 1)).todense()
    x = scaler.transform(data[-(data_size - split_index) :,: -1])
    y = one_hot.transform(data[-(data_size - split_index) :, -1].reshape(-1, 1)).todense()

    nn = mlr.NeuralNetwork(hidden_nodes=[2],
                        activation='relu',
                        algorithm='random_hill_climb',
                        max_iters=80,
                        bias=True,
                        is_classifier=True,
                        learning_rate=0.00001,
                        early_stopping=True,
                        clip_max=5,
                        restarts=1,               # random_hill_climb
                        max_attempts=50,
                        random_state=1,
                        curve=True)
    start = time.time()
    Y_acc, y_acc, loss, curve1 = train_and_run(X, Y, x, y, nn)
    t.append(time.time() - start)
    y1.append(Y_acc)
    y2.append(y_acc)
    l.append(loss)

    nn = mlr.NeuralNetwork(hidden_nodes=[2],
                        activation='tanh',
                        algorithm='simulated_annealing',
                        max_iters=25,
                        bias=True,
                        is_classifier=True,
                        learning_rate=0.00001,
                        early_stopping=True,
                        clip_max=5,
                        schedule=mlr.GeomDecay(init_temp=1, decay=0.99, min_temp=0.001), # simulated_annealing
                        max_attempts=1,
                        random_state=1,
                        curve=True)
    start = time.time()
    Y_acc, y_acc, loss, curve2 = train_and_run(X, Y, x, y, nn)
    t.append(time.time() - start)
    y1.append(Y_acc)
    y2.append(y_acc)
    l.append(loss)

    nn = mlr.NeuralNetwork(hidden_nodes=[2],
                        activation='relu',
                        algorithm='genetic_alg',
                        max_iters=80,
                        bias=True,
                        is_classifier=True,
                        learning_rate=0.00001,
                        early_stopping=True,
                        clip_max=5,
                        pop_size=15,
                        mutation_prob=0.1,
                        max_attempts=1,
                        random_state=1,
                        curve=True)
    start = time.time()
    Y_acc, y_acc, loss, curve3 = train_and_run(X, Y, x, y, nn)
    t.append(time.time() - start)
    y1.append(Y_acc)
    y2.append(y_acc)
    l.append(loss)

    nn = neural_network.MLPClassifier(hidden_layer_sizes=[2],
                                    activation='relu',
                                    solver='lbfgs',
                                    max_iter=80,
                                    alpha=0.00001,
                                    random_state=1)
    start = time.time()
    Y_acc, y_acc, loss, curve = train_and_run(X, Y, x, y, nn)
    t.append(time.time() - start)
    y1.append(Y_acc)
    y2.append(y_acc)
    l.append(nn.loss_)
    a, b, c = learning_curve(nn, np.asarray(X), np.asarray(Y))

    plt.title(f'Neural Net: Search and Back Propagation (train)')
    plt.xlabel('Weight method')
    plt.ylabel('Error rate')
    plt.bar(['RHC', 'SA', 'GA', 'BP'], y1, color=['red', 'green', 'blue', 'orange'], label='Training data')
    plt.savefig('plots/neural_train.png')
    plt.clf()
    plt.title(f'Neural Net: Search and Back Propagation (test)')
    plt.xlabel('Weight method')
    plt.ylabel('Error rate')
    plt.bar(['RHC', 'SA', 'GA', 'BP'], y2, color=['red', 'green', 'blue', 'orange'], label='Training data')
    plt.savefig('plots/neural_test.png')
    plt.clf()
    plt.title(f'Neural Net: Search and Back Propagation (Wall Clock Time)')
    plt.xlabel('Weight method')
    plt.ylabel('Wall clock time (seconds)')
    plt.bar(['RHC', 'SA', 'GA', 'BP'], t, color=['red', 'green', 'blue', 'orange'], label='Training data')
    plt.savefig('plots/neural_time.png')
    plt.clf()
    plt.title(f'Neural Net: Loss')
    plt.xlabel('Weight method')
    plt.ylabel('Loss')
    plt.bar(['RHC', 'SA', 'GA', 'BP'], l, color=['red', 'green', 'blue', 'orange'], label='Training data')
    plt.savefig('plots/neural_loss.png')
    plt.clf()

    plt.title('Neural Net: Random Hill Climbing Fitness Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    x, y = curve1.T
    plt.plot(y, x)
    plt.savefig('plots/neural_rhc.png')
    plt.clf()

    plt.title('Neural Net: Simulated Annealing Fitness Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    x, y = curve2.T
    plt.plot(y, x)
    plt.savefig('plots/neural_sa.png')
    plt.clf()

    plt.title('Neural Net: Genetic Algorithm Fitness Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    x, y = curve3.T
    plt.plot(x, y)
    plt.savefig('plots/neural_ga.png')
    plt.clf()


def get_data(file_name):
    data = []

    with open(file_name, 'r') as file:
        for line in file.readlines()[1:]:
            row = line.strip().split(',')
            data.append(row)

    data = np.array(data, dtype=np.float64)
    np.random.shuffle(data)

    return data


if __name__ == "__main__":
    run_search()
    run_neural()
