import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def create_feature_matrix(x, nb_features):
    """
    Creates feature matrix out of the input array of training data. The dimensions of the matrix are m X n, where
    m represents number of training data and n number of features.
    """
    tmp_features = []
    for deg in range(1, nb_features + 1):
        tmp_features.append(np.power(x, deg))
    return np.column_stack(tmp_features)


def process_data(filename):
    """
    Processes data from file which is given as input parameter. The data is read, shuffled and normalised.
    :param filename: path to file containing the data
    :return: x matrix, y array and number of samples
    """
    # read
    all_data = np.loadtxt(filename, delimiter=',')
    data = dict()
    data['x'] = all_data[:, :1]
    data['y'] = all_data[:, 1:]

    # shuffle
    nb_samples = data['x'].shape[0]
    indices = np.random.permutation(nb_samples)
    data['x'] = data['x'][indices]
    data['y'] = data['y'][indices]

    # normalize
    # axis = 0?
    data['x'] = (data['x'] - np.mean(data['x'], axis=0)) / np.std(data['x'], axis=0)
    data['y'] = (data['y'] - np.mean(data['y'])) / np.std(data['y'])

    return data['x'], data['y'], nb_samples


def polynomial_regression(data_x, data_y, nb_features, nb_samples, lmd):
    """
    Does polynomial regression with added L2 regularization given the parameter lambda as input of the function and
    returns the calculated hypothesis and total loss.
    """

    tf.reset_default_graph()
    data_x = create_feature_matrix(data_x, nb_features)

    X = tf.placeholder(shape=(None, nb_features), dtype=tf.float32, name='X')
    Y = tf.placeholder(shape=None, dtype=tf.float32, name='Y')
    w = tf.Variable(tf.zeros(nb_features), name='weights')
    bias = tf.Variable(0.0, name='bias')

    w_col = tf.reshape(w, (nb_features, 1), name='weights_reshaped')
    hyp = tf.add(tf.matmul(X, w_col), bias, name='hypothesis')

    Y_col = tf.reshape(Y, (-1, 1), name='Y_reshaped')
    regularizer = tf.nn.l2_loss(w_col, name='regularizer')
    # loss function using L2 Regularization
    loss = tf.abs(tf.reduce_mean(tf.square(hyp - Y_col) + lmd * regularizer), name='loss')
    opt_op = tf.train.AdamOptimizer().minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        total_loss = 0
        nb_epochs = 100                                                             # 100 training epochs
        for epoch in range(nb_epochs):

            # Stochastic Gradient Descent
            epoch_loss = 0
            for sample in range(nb_samples):
                feed = {X: data_x[sample].reshape((1, nb_features)),
                        Y: data_y[sample]}
                _, curr_loss = sess.run([opt_op, loss], feed_dict=feed)
                epoch_loss += curr_loss

            total_loss += epoch_loss
            epoch_loss /= nb_samples
            if (epoch + 1) % 10 == 0:                                               # print every 10-th
                print(
                    'Lambda: {}| Epoch: {}/{}| Avg loss: {:.7f}'.format(lmd, epoch + 1, nb_epochs, epoch_loss))

        w_val = sess.run(w)
        bias_val = sess.run(bias)
        print('w = ', w_val, 'bias = ', bias_val)
        print('total loss = ', total_loss, '\n')
        xs = create_feature_matrix(np.linspace(-2, 4, 100), nb_features)
        hyp_val = sess.run(hyp, feed_dict={X: xs})
        return xs, hyp_val, total_loss


def main():
    np.set_printoptions(suppress=True, precision=5)                       # display floating point numbers to 5 decimals

    data_x, data_y, nb_samples = process_data('data/funky.csv')

    # draw data
    plt.scatter(data_x[:, 0], data_y, c="b")
    plt.xlabel('X')
    plt.ylabel('Y')

    total = []
    for (c, lmd) in [('k', 0), ('y', 0.001), ('m', 0.01), ('c', 0.1), ('r', 1), ('g', 10), ('b', 100)]:
        xs, hyp_val, total_loss = polynomial_regression(data_x, data_y, 3, nb_samples, lmd)
        plt.plot(xs[:, 0].tolist(), hyp_val.tolist(), color=c, label='lmd={}'.format(lmd))
        total.append(total_loss)

    # first graph
    plt.xlim([-2, 4])
    plt.ylim([-3, 2])
    plt.legend()
    plt.show()

    # second graph
    plt.scatter([0, 0.001, 0.01, 0.1, 1, 10, 100], total, c="r")
    plt.xlabel('Lambda')
    plt.ylabel('Final loss function')
    plt.show()

    # visualise using tensorboard
    writer = tf.summary.FileWriter('.')
    writer.add_graph(tf.get_default_graph())
    writer.flush()


if __name__ == "__main__":
    main()

"""
Zakljucak:

Sa dodatom L2 regularizaciom, regresione krive su jako slicne za vrednosti lambda iz skupa {0, 0.001, 0.01, 0,1 i 1}
i dobro opisuju podatke. Finalne i prosecne vrednosti funkcije troska su takodje slicne, ali se njihova vrednost 
smanjuje zajedno sa parametrom lambda. Situacija je drugacija za parametar lambda iz skupa {10, 100}. Kako se lambda 
povecava regresiona kriva odstupa od svog optimalnog polozaja i ne opisuje dobro date podatke. Pored toga, finalne i
prosecne vrednosti funkcije troska se znacajno povecavaju sa povecanjem lambda. Stoga, trosak je proporcionalan 
vrednoscu parametra lambda, pa je optimalno uzeti 0 za vrednost lambda.

"""
