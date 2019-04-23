# TODO popuniti kodom za problem 3a

import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# np.set_printoptions(threshold=sys.maxsize)


class KNN:

    def __init__(self, nb_features, nb_classes, data, k):
        self.nb_features = nb_features
        self.nb_classes = nb_classes
        self.data = data
        self.k = k

        # Gradimo model, X je matrica podataka a Q je vektor koji predstavlja upit.
        self.X = tf.placeholder(shape=(None, nb_features), dtype=tf.float32)
        self.Y = tf.placeholder(shape=(None), dtype=tf.int32)
        self.Q = tf.placeholder(shape=(nb_features), dtype=tf.float32)

        # Racunamo kvadriranu euklidsku udaljenost i uzimamo minimalnih k.
        dists = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.X, self.Q)),
                                      axis=1))
        _, idxs = tf.nn.top_k(-dists, self.k)

        self.classes = tf.gather(self.Y, idxs)
        self.dists = tf.gather(dists, idxs)

        self.classes_one_hot = tf.one_hot(self.classes, nb_classes)
        self.scores = tf.reduce_sum(self.classes_one_hot, axis=0)

        # Klasa sa najvise glasova je hipoteza.
        self.hyp = tf.argmax(self.scores)

    # Ako imamo odgovore za upit racunamo i accuracy.
    def predict(self, query_data):

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            nb_queries = query_data['x'].shape[0]

            matches = 0
            predicted = []
            for i in range(nb_queries):
                hyp_val = sess.run(self.hyp, feed_dict={self.X: self.data['x'],
                                                        self.Y: self.data['y'],
                                                        self.Q: query_data['x'][i]})
                predicted.append(hyp_val)

                if query_data['y'] is not None:
                    actual = query_data['y'][i]
                    match = (hyp_val == actual)
                    if match:
                        matches += 1
                    # if i % 10 == 0:
                    print('Test example: {}/{}| Predicted: {}| Actual: {}| Match: {}'
                          .format(i + 1, nb_queries, hyp_val, actual, match))

            accuracy = matches / nb_queries
            # print('{} matches out of {} examples'.format(matches, nb_queries))
            return accuracy, predicted


filename = 'data/iris.csv'
mapper_key_val = {
    0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'
}
mapper_val_key = {v: k for k, v in mapper_key_val.items()}

all_data = np.loadtxt(filename, delimiter=',', dtype=str, skiprows=1)
all_data = np.array([np.array([item[0], item[1], item[2], item[3], mapper_val_key[item[4]]]).astype(float) for item in all_data])

nb_features = 2 # prva dva featurea
nb_classes = 3 # 3 vrste cveta perunike
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

# Pokrecemo kNN na test skupu.
k = 3
train_data = {'x': data_train['x'], 'y': data_train['y']}
knn = KNN(nb_features, nb_classes, train_data, k)
accuracy, predicted = knn.predict({'x': data_test['x'], 'y': data_test['y']})
print('Test set accuracy: ', accuracy)

print('Building map...')
# Generisemo grid.
step_size = 0.01
x1, x2 = np.meshgrid(np.arange(min(data['x'][:, 0]), max(data['x'][:, 0]),
                               step_size),
                     np.arange(min(data['x'][:, 1]), max(data['x'][:, 1]),
                               step_size))
x_feed = np.vstack((x1.flatten(), x2.flatten())).T
# Racunamo vrednost hipoteze.
accuracy, pred_val = knn.predict({'x': x_feed, 'y': None})
# pred_val = sess.run(pred, feed_dict={X: x_feed})

pred_val = np.array(pred_val)
pred_plot = pred_val.reshape([x1.shape[0], x1.shape[1]])

# Crtamo contour plot.
from matplotlib.colors import LinearSegmentedColormap

classes_cmap = LinearSegmentedColormap.from_list('classes_cmap',
                                                 ['lightblue',
                                                  'lightgreen',
                                                  'lightyellow'])
plt.contourf(x1, x2, pred_plot, cmap=classes_cmap, alpha=0.7)

# Crtamo sve podatke preko.
idxs_0 = train_data['y'] == 0.
idxs_1 = train_data['y'] == 1.
idxs_2 = train_data['y'] == 2.
plt.scatter(train_data['x'][idxs_0, 0], train_data['x'][idxs_0, 1], c='b',
            edgecolors='k', label=mapper_key_val[0])
plt.scatter(train_data['x'][idxs_1, 0], train_data['x'][idxs_1, 1], c='g',
            edgecolors='k', label=mapper_key_val[1])
plt.scatter(train_data['x'][idxs_2, 0], train_data['x'][idxs_2, 1], c='y',
            edgecolors='k', label=mapper_key_val[2])
plt.legend()

plt.show()