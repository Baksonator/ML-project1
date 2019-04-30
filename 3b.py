# TODO popuniti kodom za problem 3b
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

_3a = __import__('3a')


def get_best_k(should_draw=True):
    filename = 'data/iris.csv'
    mapper_key_val = {
        0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'
    }
    mapper_val_key = {v: k for k, v in mapper_key_val.items()}

    all_data = np.loadtxt(filename, delimiter=',', dtype=str, skiprows=1)
    all_data = np.array(
        [np.array([item[0], item[1], item[2], item[3], mapper_val_key[item[4]]]).astype(float) for item in all_data])

    nb_features = 2  # prva dva featurea
    nb_classes = 3  # 3 vrste cveta perunike
    data = dict()
    data['x'] = all_data[:, :nb_features]
    data['y'] = all_data[:, 4]

    # Mesanje.
    nb_samples = data['x'].shape[0]
    indices = np.random.permutation(nb_samples)
    data['x'] = data['x'][indices]
    data['y'] = data['y'][indices]

    # Normalizacija
    data['x'] = (data['x'] - np.mean(data['x'], axis=0)) / np.std(data['x'], axis=0)

    training_ratio = 0.8
    test_ratio = 0.2

    nb_train = int(training_ratio * nb_samples)
    data_train = dict()
    data_train['x'] = data['x'][:nb_train]
    data_train['y'] = data['y'][:nb_train]

    nb_test = nb_samples - nb_train
    data_test = dict()
    data_test['x'] = data['x'][nb_train:]
    data_test['y'] = data['y'][nb_train:]

    accuracies = list()

    for k in range(1, 16):
        train_data = {'x': data_train['x'], 'y': data_train['y']}
        knn = _3a.KNN(nb_features, nb_classes, train_data, k)
        accuracy, predicted = knn.predict({'x': data_test['x'], 'y': data_test['y']})
        accuracies.append(accuracy)

    if should_draw:
        print(accuracies)
        plt.plot([x for x in range(1, 16)], accuracies)
        plt.axis([0, 16, 0.5, 1])
        plt.grid(True)
        plt.show()

    max_accuracy = max(accuracies)
    return [ind for ind, x in enumerate(accuracies) if x == max_accuracy]


get_best_k()

# # Najbolje K
# k_max = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# for i in range(0, 50):
#     best_k = get_best_k(should_draw=False)
#     for index in best_k:
#         k_max[index] += 1
# print(k_max)

# U ovom primeru, ukljucujuci 2 feature-a i iteracijom kroz k od 1 do 15,
# nema znacajne razlike, jer ukljucujemo samo mali broj feature-a.
# Vidimo ucestale oscilacije koje su izmedju 0.6 i 0.8, u retkim slucajevima accuracy prelazi preko 0.8.
# U vecini ovakvih slucajeva je bolje uzeti K vece od 2.
# Prilicno random.
