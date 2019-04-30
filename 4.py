from nltk import FreqDist
from nltk.tokenize import wordpunct_tokenize
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from random import shuffle
import numpy as np
import math
import re
import glob
import heapq
# nltk.download() # If it's you first time running, you have to run this line as well


class MultinomialNaiveBayes:
    def __init__(self, nb_classes, nb_words, pseudocount):
        self.nb_classes = nb_classes # number of classes
        self.nb_words = nb_words # number of words in vocab
        self.pseudocount = pseudocount # Laplace smoothing constant

    def fit(self, data):
        """
        Fits the model to the training data given
        :param data: Feature vectors (data['x']) and labels (data['y'])
        :return:
        """

        x, labels = data['x'], data['y']
        nb_examples = len(labels)

        # Calculate class priors
        self.priors = np.bincount(labels) / nb_examples

        # Sum event scores for each class
        scores = np.zeros((self.nb_classes, self.nb_words))
        for i in range(nb_examples):
            c = labels[i]
            for w, cnt in x[i].items():
                scores[c][w] += cnt

        # Calculate event likelihoods for each class
        self.likelihoods = np.zeros((self.nb_classes, self.nb_words))
        for c in range(self.nb_classes):
            for w in range(self.nb_words):
                up = scores[c][w] + self.pseudocount
                down = np.sum(scores[c]) + self.nb_words * self.pseudocount
                self.likelihoods[c][w] = up / down

    def predict(self, xs):
        """
        Predicts a class based on the model and new data
        :param xs: List of feature vectors
        :return: List of predictions for given feature vectors
        """

        nb_examples = len(xs)
        preds = []
        for i in range(nb_examples):
            # Calculate log probabilities for each class
            log_probs = np.zeros(self.nb_classes)
            for c in range(self.nb_classes):
                log_prob = np.log(self.priors[c])
                for w, cnt in xs[i].items():
                    log_prob += cnt * np.log(self.likelihoods[c][w])
                log_probs[c] = log_prob

            # Max log probability gives the prediction
            pred = np.argmax(log_probs)
            preds.append(pred)
        return preds


class Data:
    def __init__(self, path, nb_data, train_pct, test_pct, nb_words):
        self.path = path
        self.nb_data = nb_data
        self.nb_train = int(train_pct * nb_data)
        self.nb_test = int(test_pct * nb_data)
        self.nb_words = nb_words
        self.load_data()
        self.clean_data()
        # self.featurize_bow() # BoW
        self.featurize_tfidf() # TF-IDF

    def load_data(self):
        """
        Loads all the data from files, data is kept separately for easier data distribution train and test set
        :return:
        """

        self.positive_corpus = []
        self.negative_corpus = []

        # Lists of all positive and all negative reviews (files)
        print('Loading the corpus...')
        all_positive_files = glob.glob(self.path + 'pos/*.txt')
        all_negative_files = glob.glob(self.path + 'neg/*.txt')

        # Add all of the docs into the corpus
        for file in all_positive_files:
            with open(file, 'r', encoding='utf8') as fd:
                self.positive_corpus.append(fd.read())

        for file in all_negative_files:
            with open(file, 'r', encoding='utf8') as fd:
                self.negative_corpus.append(fd.read())

    def clean_data(self):
        """
        Cleans the data set from stopwords and special characters, coverts to lower case and stemms all the words
        :return:
        """

        print('Cleaning the corpus...')
        porter = PorterStemmer()
        self.clean_corpus_positive = []
        self.clean_corpus_negative = []
        stop_punc = set(stopwords.words('english')).union(set(punctuation))

        for doc in self.positive_corpus:
            words = wordpunct_tokenize(doc)
            words_lower = [w.lower() for w in words]
            words_remspecial = [re.sub(r'[^a-zA-Z0-9\s]', '', w) for w in words_lower]
            words_filtered = [w for w in words_remspecial if w not in stop_punc]
            words_stemmed = [porter.stem(w) for w in words_filtered]
            words_remempty = list(filter(None, words_stemmed))
            words_rembr = list(filter(lambda a: a != 'br', words_remempty))
            self.clean_corpus_positive.append(words_rembr)

        for doc in self.negative_corpus:
            words = wordpunct_tokenize(doc)
            words_lower = [w.lower() for w in words]
            words_remspecial = [re.sub(r'[^a-zA-Z0-9\s]', '', w) for w in words_lower]
            words_filtered = [w for w in words_remspecial if w not in stop_punc]
            words_stemmed = [porter.stem(w) for w in words_filtered]
            words_remempty = list(filter(None, words_stemmed))
            words_rembr = list(filter(lambda a: a != 'br', words_remempty))
            self.clean_corpus_negative.append(words_rembr)

    def featurize_bow(self):
        """
        Create a BoW feature vector for each doc, divide data into train and test set
        :return:
        """

        # Use only nb_words most frequent words for efficiency purposes
        print('Calculating BoW...')
        freq = FreqDist([w for doc in (self.clean_corpus_positive + self.clean_corpus_negative) for w in doc])
        best_words, _ = zip(*freq.most_common(self.nb_words))

        x_positive = []
        x_negative = []

        # A BoW for each doc, sparse representation
        for doc in self.clean_corpus_positive:
            bow = dict()
            for i in range(self.nb_words):
                cnt = doc.count(best_words[i])
                if cnt > 0:
                    bow[i] = cnt
            x_positive.append(bow)

        for doc in self.clean_corpus_negative:
            bow = dict()
            for i in range(self.nb_words):
                cnt = doc.count(best_words[i])
                if cnt > 0:
                    bow[i] = cnt
            x_negative.append(bow)

        # Shuffle the data for randomness
        shuffle(x_positive)
        shuffle(x_negative)

        # Class labels, 1 for positive, 0 for negative
        positive_labels = np.ones(self.nb_data, dtype=int)
        negative_labels = np.zeros(self.nb_data, dtype=int)

        self.train_set = {
            'x': (x_positive[self.nb_test:] + x_negative[self.nb_test:]),
            'y': (np.concatenate([positive_labels[self.nb_test:], negative_labels[self.nb_test:]]))
        }

        self.test_set = {
            'x': (x_positive[:self.nb_test] + x_negative[:self.nb_test]),
            'y': (np.concatenate([positive_labels[:self.nb_test], negative_labels[:self.nb_test]]))
        }

    def featurize_tfidf(self):
        """
        Create a TF-IDF feature vector for each doc, divide data into train and test set
        :return:
        """

        # Use only nb_words most frequent words for efficiency purposes
        print('Calculating TF-IDF...')
        freq = FreqDist([w for doc in (self.clean_corpus_positive + self.clean_corpus_negative) for w in doc])
        best_words, _ = zip(*freq.most_common(self.nb_words))

        x_positive = []
        x_negative = []

        # Get count of each word across the whole corpus, single out most frequent words in positive and negative corpus
        doc_counts = dict()
        doc_counts_positive = dict()
        doc_counts_negative = dict()
        for word in best_words:
            doc_counts[word] = 0
            doc_counts_positive[word] = 0
            doc_counts_negative[word] = 0
            for doc in self.clean_corpus_positive:
                if word in doc:
                    doc_counts[word] += 1
                    doc_counts_positive[word] += 1
            for doc in self.clean_corpus_negative:
                if word in doc:
                    doc_counts[word] += 1
                    doc_counts_negative[word] += 1
        best_words_positive = heapq.nlargest(5, doc_counts_positive, key=doc_counts_positive.get)
        print('Most frequent positive words:')
        print(best_words_positive)
        best_words_negative = heapq.nlargest(5, doc_counts_negative, key=doc_counts_negative.get)
        print('Most frequent negative words:')
        print(best_words_negative)

        # Calculate the LR metric for all words that are viable
        lr = dict()
        for word in best_words:
            if doc_counts_positive[word] >= 10 and doc_counts_negative[word] >= 10:
                lr[word] = doc_counts_positive[word] / doc_counts_negative[word]
        highest_lr = heapq.nlargest(5, lr, key=lr.get)
        print('Highest LR words:')
        print(highest_lr)
        lowest_lr = heapq.nsmallest(5, lr, key=lr.get)
        print('Lowest LR words:')
        print(lowest_lr)

        # Calculate idf value for each word
        idf_table = dict()
        for word in best_words:
            idf = math.log10((len(self.clean_corpus_positive) + len(self.clean_corpus_negative)) / doc_counts[word])
            idf_table[word] = idf

        def tfidf_score(word, doc):
            tf = doc.count(word) / len(doc)
            idf = idf_table[word]
            return tf * idf

        # A TF-IDF vector for each doc, sparse representation
        for doc in self.clean_corpus_positive:
            tfidf = dict()
            for i in range(self.nb_words):
                score = tfidf_score(best_words[i], doc)
                if score > 0:
                    tfidf[i] = score
            x_positive.append(tfidf)

        for doc in self.clean_corpus_negative:
            tfidf = dict()
            for i in range(self.nb_words):
                score = tfidf_score(best_words[i], doc)
                if score > 0:
                    tfidf[i] = score
            x_negative.append(tfidf)

        # Shuffle the data for randomness
        shuffle(x_positive)
        shuffle(x_negative)

        # Class labels, 1 for positive, 0 for negative
        positive_labels = np.ones(self.nb_data, dtype=int)
        negative_labels = np.zeros(self.nb_data, dtype=int)

        self.train_set = {
            'x': (x_positive[self.nb_test:] + x_negative[self.nb_test:]),
            'y': (np.concatenate([positive_labels[self.nb_test:], negative_labels[self.nb_test:]]))
        }

        self.test_set = {
            'x': (x_positive[:self.nb_test] + x_negative[:self.nb_test]),
            'y': (np.concatenate([positive_labels[:self.nb_test], negative_labels[:self.nb_test]]))
        }


k = 5 # k-fold cross-validation, if set to 1 then normal training

data = Data('data/imdb/', 1250, 0.8, 0.2, 10000)
train_set = data.train_set
test_set = data.test_set

# Randomize the train set data because of cross validation
train_set['x'] = np.array(train_set['x'])
indices = np.random.permutation(len(train_set['x']))
train_set['x'] = train_set['x'][indices]
train_set['y'] = train_set['y'][indices]

# Train various models using cross-validation
print('Training the model...')
models = dict()
num = len(train_set['x']) // k
for lp in range(1, 11):
    avg_accuracy = 0
    for i in range(k):
        validate_set = {
            'x': (train_set['x'][i * num:(i + 1) * num]),
            'y': (train_set['y'][i * num:(i + 1) * num])
        }
        real_train_set = {
            'x': (np.concatenate([train_set['x'][:i * num], train_set['x'][:(i + 1) * num]])),
            'y': (np.concatenate([train_set['y'][:i * num], train_set['y'][:(i + 1) * num]]))
        }
        model = MultinomialNaiveBayes(2, data.nb_words, lp)
        model.fit(real_train_set)

        # print('Predicting validate set...')
        predictions_train = model.predict(validate_set['x'])
        nb_correct = 0
        nb_total = len(predictions_train)
        for j in range(nb_total):
            if validate_set['y'][j] == predictions_train[j]:
                nb_correct += 1
        accuracy = nb_correct / nb_total
        avg_accuracy += accuracy
    avg_accuracy /= k
    print('Accuracy for parameter ' + str(lp) + ' is ' + str(avg_accuracy))
    models[lp] = avg_accuracy

# Pick best model and train it on whole train set
best_model_num = max(models, key=models.get)
model = MultinomialNaiveBayes(2, data.nb_words, best_model_num)
model.fit(train_set)

print('Predicting train set...')
predictions = model.predict(train_set['x'])
nb_correct = 0
nb_total = len(predictions)
for i in range(nb_total):
    if train_set['y'][i] == predictions[i]:
        nb_correct += 1
accuracy = nb_correct / nb_total
print('Accuracy on train set is:', accuracy)

confusion_matrix = np.zeros(shape=(2, 2)) # [[TN, FP], [FN, TP]]

print('Predicting test set...')
predictions = model.predict(test_set['x'])
nb_correct = 0
nb_total = len(predictions)
for i in range(nb_total):
    if test_set['y'][i] == predictions[i]:
        nb_correct += 1
    if test_set['y'][i] == 1:
        if predictions[i] == 1:
            confusion_matrix[1][1] += 1
        else:
            confusion_matrix[1][0] += 1
    else:
        if predictions[i] == 1:
            confusion_matrix[0][1] += 1
        else:
            confusion_matrix[0][0] += 1
accuracy = nb_correct / nb_total
print('Accuracy on test set is:', accuracy)

print('Confusion matrix:')
print(confusion_matrix)

"""
5 najcesce koriscenih reci u pozitivnim kritikama su: film, movi, one, like, time
5 najcesce koriscenih reci u negativnim kritikama su: movi, film, one, like, time
Primecujemo da su iste reci najcesce i u pozitivnim i u negativnim kritikama, jedina razlika jeste broj pojavljivanja,
tj. redosled. Ovo znaci da nam ove reci uopste nisu bitne za resavanje naseg problema, tj. mozemo ih i izostaviti.
Buduci da koristimo TF-IDF, njihov score ce bas iz ovog razloga biti manji, pa ih ne moramo zaista izbacivati.
5 reci sa najvisom vrednoscu LR metrike su: perfectli, strong, delight, highli, excel
5 reci sa najnizom vrednoscu LR metrike su: worst, stupid, wast, crap, ridicul
Ove reci su veoma korisne za nas cilj klasifikacije, njihovo pojavljivanje u nekom buducem doc-u je snazan pokazatelj
da je ta kritika pozitivna/negativna. Predstavljaju suprotnost reci koje smo pronasli u prethodnoj analizi, 
TF-IDF ce im dodeljivati veci score i samim tim imace vecu tezinu u nasoj klasifikaciji. Reci koje smo pronasli gore ce
imati LR vrednosti blizu 1, odnosno nalazice se u sredini raspodele. LR metrika je znacajna jer nam pokazuje koje reci
imaju vece znacenje (tezinu) u nasem kontekstu, a koje manju, odnosno govori nam na koje reci treba obratiti posebnu 
paznju a koje reci treba zanemariti.
"""
