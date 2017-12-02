import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean as metric
import scipy.stats


def separate_dataset(dataset, ratio):
    train_size = int(len(dataset) * ratio)

    indices = np.random.permutation(len(dataset))

    training_indices = indices[:train_size]
    test_indices = indices[train_size:]

    training = dataset[training_indices, :]
    test = dataset[test_indices, :]

    return training, test


def join_datasets(set_1, set_2):
    return np.vstack((set_1, set_2)).astype(np.float32)


def accuracy(prericted, real):
    correct = sum(prericted == real)
    total = len(prericted)

    return 100.0 * correct / total


def calculate_distancies(data, unknown):
    data_size = data.shape[0]
    unknown_size = unknown.shape[0]

    dist = np.zeros((unknown_size, data_size))

    for i in range(unknown_size):
        for j in range(data_size):
            dist[i, j] = metric(unknown[i], data[j])

    return dist


def predict(distances, data_y, k):
    distances_size = distances.shape[0]
    y_prediction = np.zeros(distances_size)

    for i in range(distances_size):
        distance = distances[i]
        y_closest = data_y[distance.argsort()[:k]]
        y_prediction[i] = scipy.stats.mode(y_closest).mode

    return y_prediction


def compare_k(data_x, data_y, test_x, test_y, kmin=1, kmax=50, kstep=4):
    '''
        Main comparing function
    '''
    k = list(range(kmin, kmax, kstep))
    steps = len(k)
    features = np.zeros((steps, 3))

    print('Evaluating distancies started')

    distancies = calculate_distancies(data_x, test_x)
    miss = []
    s1 = data_x.shape[0]
    s2 = test_x.shape[0]

    for j in range(steps):
        yk = predict(distancies, data_y, k[j])
        features[j][0] = k[j]
        features[j][1] = accuracy(yk, test_y)
        cond = yk != test_y
        miss.append({
            'k': k[j],
            'acc': features[j][1],
            'x': test_x[cond]}
        )

        print('k={0}, accuracy = {1}%, time = {2} sec'.format(k[j], features[j][1], features[j][2]))

    return features, miss

training_set_ratio = 0.67

num_observations = 300
x1 = np.random.multivariate_normal([1, 6], [[2.2, .65], [.75, 1]], num_observations)
x2 = np.random.multivariate_normal([-2, 2], [[3.2, .75], [.75, 2]], num_observations)

X = np.vstack((x1, x2)).astype(np.float32)
Y = np.hstack((np.zeros(num_observations),
               np.ones(num_observations)))



plt.figure()
plt.scatter(x1[:, 0], x1[:, 1], color="red")
plt.scatter(x2[:, 0], x2[:, 1], color="blue")

plt.show()
