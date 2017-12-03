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


def generate_datasets(amount_of_observations, training_set_ratio):
    x1 = np.random.multivariate_normal([1, 6], [[2.2, .65], [.75, 1]], amount_of_observations)
    x2 = np.random.multivariate_normal([-2, 2], [[1.2, .75], [.75, 2]], amount_of_observations)

    train_set_1, test_set_1 = separate_dataset(x1, training_set_ratio)
    train_set_2, test_set_2 = separate_dataset(x2, training_set_ratio)

    x = join_datasets(train_set_1, train_set_2)

    y = np.hstack((np.zeros(len(train_set_1)),
                   np.ones(len(train_set_2))))

    return x1, x2, x, y, {0: test_set_1, 1: test_set_2}


def join_datasets(set_1, set_2):
    return np.vstack((set_1, set_2)).astype(np.float32)


def accuracy(predicted, non_predicted):
    correct = len(predicted)
    total = len(predicted) + len(non_predicted)

    return 100.0 * correct / total


def calculate_distances(data, unknown):
    data_size = data.shape[0]

    dist = np.zeros(data_size)

    for j in range(data_size):
        dist[j] = metric(data[j], unknown)

    return dist


def predict(train_set, class_ids, test_set, k):
    prediction_0 = []
    prediction_1 = []

    for id, vector in test_set.items():
        for i in range(len(vector)):
            distances = calculate_distances(train_set, vector[i])

            paired_set = np.array([[distances[j], class_ids[j]] for j in np.argsort(distances)])

            new_set = paired_set[-k:]
            pred = np.abs(1 - int(scipy.stats.mode(new_set[:, 1]).mode))

            if id == 0:
                prediction_0.append(pred)
            else:
                prediction_1.append(pred)

    return {0: np.array(prediction_0),
            1: np.array(prediction_1)}


def generate_prediction(train_set, class_ids, test_set, k):
    prediction = predict(train_set, class_ids, test_set, k)

    true_pred = []
    false_pred = []

    for id, pred in prediction.items():
        for i in range(len(pred)):
            if id == pred[i]:
                true_pred.append(test_set[id][i])
            else:
                false_pred.append(test_set[id][i])

    return np.array(true_pred), np.array(false_pred)


first_class, second_class, train_set, class_ids, test_set = generate_datasets(amount_of_observations=400,
                                                                              training_set_ratio=0.67)

true_prediction, false_prediction = generate_prediction(train_set, class_ids, test_set, k=15)

print("Accuracy rate: {}%".format(accuracy(true_prediction, false_prediction)))

plt.figure()
plt.scatter(first_class[:, 0], first_class[:, 1], color="blue", label="Zero class")
plt.scatter(second_class[:, 0], second_class[:, 1], color="cyan", label="One class")

plt.scatter(true_prediction[:, 0], true_prediction[:, 1], color="green", s=4, label="True prediction")
plt.scatter(false_prediction[:, 0], false_prediction[:, 1], color="red", label="False prediction")

plt.legend()
plt.show()