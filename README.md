The algorithms can be run by doing the following:
Polynomial regression:

2a.py will run plain polynomial regression with 6 different polynomial degrees (1 to 6) for the data set funky.csv. The first graph produced shows all the data in 2D as well as all 6 of the polynomial curves, and the second graph shows the dependency of the cost functions from the degree of the polynomial.

2b.py will run polynomial regression with degree 3 and L2 regularization. 7 different values for the lambda parameter were used. The graphs produced are the same as in 2a, just with the variable being lambda instead of the degree of the polynomial.

k-NN:

3a.py will run k-NN with regard to only the first 2 parameters of iris.csv, with parameter k = 3. It will produce as output the accuracy of the algorithm, and a 2D graph which shows all the data coloured with respect to the class they have been put into. The graph will also show the areas for each class by colouring the background.

3b.py will run k-NN with regard to only the first 2 parameters of iris.csv, with parameter k taking values from the range of integers [1,15]. It will produce the accuracy for each k, and a graph showing the dependency of k and the accuracy.

3c.py will do the same as 3b.py but will take into regard all of the parameters of iris.csv.

Naive Bayes:

4.py will run do Naive Bayes classification for positive and negative imdb reviews. It will output the accuracy of the model, the confusion matrix and other useful metrics.
