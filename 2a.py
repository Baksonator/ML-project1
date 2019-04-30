import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


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
    data['x'] = (data['x'] - np.mean(data['x'], axis=0)) / np.std(data['x'], axis=0)
    data['y'] = (data['y'] - np.mean(data['y'])) / np.std(data['y'])

    return data['x'], data['y'], nb_samples


def polynomial_regression(data_x, data_y, nb_features, nb_samples):
    """
    Does polynomial regression of degree given as parameter of the function and returns the calculated hypothesis
    and total loss.
    """

    tf.reset_default_graph()
    data_x = create_feature_matrix(data_x, nb_features)

    X = tf.placeholder(shape=(None, nb_features), dtype=tf.float32)
    Y = tf.placeholder(shape=None, dtype=tf.float32)
    w = tf.Variable(tf.zeros(nb_features))
    bias = tf.Variable(0.0)

    w_col = tf.reshape(w, (nb_features, 1))
    hyp = tf.add(tf.matmul(X, w_col), bias)

    Y_col = tf.reshape(Y, (-1, 1))
    loss = tf.reduce_mean(tf.square(hyp - Y_col))
    opt_op = tf.train.AdamOptimizer().minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        nb_epochs = 100                                                         # 100 training epochs
        for epoch in range(nb_epochs):

            # Stochastic Gradient Descent
            epoch_loss = 0
            for sample in range(nb_samples):
                feed = {X: data_x[sample].reshape((1, nb_features)),
                        Y: data_y[sample]}
                _, curr_loss = sess.run([opt_op, loss], feed_dict=feed)
                epoch_loss += curr_loss

            epoch_loss /= nb_samples
            if (epoch + 1) % 10 == 0:                                           # print every 10-th
                print('Degree: {}| Epoch: {}/{}| Avg loss: {:.7f}'.format(nb_features, epoch + 1, nb_epochs, epoch_loss))

        w_val = sess.run(w)
        bias_val = sess.run(bias)
        print('w = ', w_val, 'bias = ', bias_val)
        xs = create_feature_matrix(np.linspace(-2, 4, 100), nb_features)
        hyp_val = sess.run(hyp, feed_dict={X: xs})
        final_loss = sess.run(loss, feed_dict={X: data_x, Y: data_y})
        print('final loss = ', final_loss, '\n')
        return xs, hyp_val, final_loss


def main():
    np.set_printoptions(suppress=True, precision=5)                     # display floating point numbers to 5 decimals

    data_x, data_y, nb_samples = process_data('data/funky.csv')

    # draw data
    plt.scatter(data_x[:, 0], data_y, c="b")
    plt.xlabel('X')
    plt.ylabel('Y')

    final = []
    nb_features = 1
    for c in 'kymcrg':
        xs, hyp_val, final_loss = polynomial_regression(data_x, data_y, nb_features, nb_samples)
        plt.plot(xs[:, 0].tolist(), hyp_val.tolist(), color=c, label='deg(p)={}'.format(nb_features))
        final.append(final_loss)
        nb_features += 1

    # first graph
    plt.xlim([-2, 4])
    plt.ylim([-3, 2])
    plt.legend()
    plt.show()

    # second graph
    plt.scatter([1, 2, 3, 4, 5, 6], final, c="r")
    plt.xlabel('Degree of polynomial')
    plt.ylabel('Final loss function')
    plt.show()


if __name__ == "__main__":
    main()

"""
Zakljucak:

Regresiona kriva i funkcija troska su jako slicne za polinome stepena 1 i 2. Tada je vrednost finalne funkcije troska 
na celom skupu ~0.31352472, dok je prosecan trosak ~0.3140487. Krive prvog i drugog stepena ne "fituju" podatke dovoljno
precizno, sto nije slucaj sa polinomima veceg stepena. Regresiona kriva treceg stepena prati podatke dobro, vrednost
finalne funkcije troska je ~0.108210765, a prosecan trosak iznosi ~0.1094187. Slican je slucaj sa polinomima stepena 4, 
5 i 6. Kako se stepen polinoma povecava, regresiona kriva se ne menja znacajno. Takodje se finalne i prosecne funkcija 
troska ne smanjuju. Stoga, najoptimalnije je izbrati stepen polinoma 3. Sa samog grafika koji prikazuje podatke mozemo 
primetiti da polinomijalna kriva jeste stepena 3, jer postoje dva "skretanja".

"""
