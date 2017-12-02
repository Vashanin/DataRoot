import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean as metric


def dataset_generator(size):
    return np.random.random((size, 2))


def get_initial_centroids(dataset, k):
    return np.random.permutation(dataset.copy())[:k]


def define_closest(dataset, centroids):
    closest = []

    for point in dataset:
        closest_centroid = None
        dist_to_closest_centroid = None

        for i in range(len(centroids)):
            distance = metric(point, centroids[i])

            if (closest_centroid is None) or (dist_to_closest_centroid > distance):
                closest_centroid = i
                dist_to_closest_centroid = distance

        closest.append(closest_centroid)

    return np.array(closest)


def move_centroid(dataset, closest):
    k = len(np.unique(closest))
    new_centroids = np.zeros((k, 2))

    for i in range(len(dataset)):
        centroid = int(closest[i])
        new_centroids[closest[i]] += dataset[i] / len(closest[closest == centroid])

    return new_centroids


def do_iteration(dataset, centroids):
    closest = define_closest(dataset, centroids)

    return move_centroid(dataset, closest)


def main(k, amount_of_iterations):
    dset = dataset_generator(500)

    initial_centroids = get_initial_centroids(dset, k)

    new_centroids = initial_centroids
    for i in range(amount_of_iterations):
        new_centroids = move_centroid(dset, define_closest(dset, new_centroids))

    plt.figure()

    plt.subplot(121)
    plt.subplots_adjust(left=0.04, right=0.99, top=0.92, bottom=0.1)

    plt.title("Initial centroids")
    plt.scatter(dset[:, 0],
                dset[:, 1], color="blue")
    plt.scatter(initial_centroids[:, 0],
                initial_centroids[:, 1], color="red", s=100)

    plt.subplot(122)

    plt.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.1)

    plt.title("Final centroids")
    plt.scatter(dset[:, 0],
                dset[:, 1], color="blue")

    plt.scatter(new_centroids[:, 0],
                new_centroids[:, 1], color="red", s=100)

    plt.show()


main(k=5, amount_of_iterations=20)
