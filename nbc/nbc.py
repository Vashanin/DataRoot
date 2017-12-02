import numpy as np


def split_dataset(dataset, ratio):
    train_size = int(len(dataset) * ratio)

    indices = np.random.permutation(dataset.shape[0])
    
    training_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    training = dataset[training_indices, :]
    test = dataset[test_indices, :]

    return training, test


def density_function(x, mean, std):
    return np.exp(-(x - mean)**2 / (2 * std**2)) / (np.sqrt(2 * np.pi) * std)


def divide_on_classes(dataset):
    zero_class = []
    one_class = []

    for row in dataset:
        if row[-1] == 0:
            zero_class.append(row)
        if row[-1] == 1:
            one_class.append(row)

    return zero_class, one_class


def dist_params(dataset):
    means = []
    stds = []

    for column in dataset:
        means.append(np.mean(column[:-1]))
        stds.append(np.std(column[:-1], ddof=1))

    return means, stds


def class_dist_params(dataset):
    separated_classes = divide_on_classes(dataset)

    class_params = []

    for i in range(len(separated_classes)):
        class_params.append(
            dist_params(separated_classes[i])
        )

    return class_params


def count_class_probabilities(class_params, input_value):
    probabilities = {}

    for i in range(len(class_params)):
        means = class_params[i][0][0]
        stds = class_params[i][1][0]

        probabilities[i] = np.prod(
            density_function(input_value[:-1], means, stds)
        )

    return probabilities


def make_prediction(class_params, input_value):
    probabilities = count_class_probabilities(class_params, input_value)

    best_class = None
    best_probability = -1

    for class_label, probability in probabilities.items():
        if class_label is None or probability > best_probability:
            best_probability = probability
            best_class = class_label

    return best_class


def get_predictions(class_params, test_set):
    predictions = []

    for vector in test_set:
        predictions.append(
            make_prediction(class_params, vector)
        )

    return predictions


def get_data(filename):
    return np.genfromtxt(filename, delimiter=",")


def get_accuracy(test_set, predictions):
    correct_predictions = 0

    for i in range(len(test_set)):
        if test_set[i][-1] == predictions[i]:
            correct_predictions += 1

    return 100 * correct_predictions / len(test_set)


def main():
    filename = "data.csv"

    training_set_ratio = 0.67

    dataset = get_data(filename)

    training_set, test_set = split_dataset(dataset, training_set_ratio)

    class_params = class_dist_params(training_set)

    predictions = get_predictions(class_params, test_set)

    accuracy = get_accuracy(test_set, predictions)

    print("Length of training set: {}".format(len(training_set)))
    print("Length of test set: {}".format(len(test_set)))
    print("\nAccuracy of NBC prediction is {}%".format(accuracy))

main()