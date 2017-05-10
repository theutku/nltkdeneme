import random
from statistics import mode

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI


class VoteClassifier(ClassifierI):

    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            vote = c.classify(features)
            votes.append(vote)

        try:
            md = mode(votes)
            return md
        except:
            return 'Classification Draw!'

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            vote = c.classify(features)
            votes.append(vote)

        try:
            choice_votes = votes.count(mode(votes))
            confid = float(choice_votes) / len(votes)
            return confid
        except:
            return 'Confidence Draw!'


# documents = []

# for category in movie_reviews.categories():
#     for fileid in movie_reviews.fileids(category):
#         documents.append((list(movie_reviews.words(fileid)), category))

# random.shuffle(documents)
# # print(documents[1])

# all_words = []

# for word in movie_reviews.words():
#     all_words.append(word.lower())

documents = []

short_pos = open('short_reviews/positive.txt', 'rb').read()
short_neg = open('short_reviews/negative.txt', 'rb').read()

short_pos = short_pos.decode('utf8', 'ignore')
short_neg = short_neg.decode('utf8', 'ignore')

for review in short_pos.split('\n'):
    words = word_tokenize(review)
    documents.append((words, 'pos'))

for review in short_neg.split('\n'):
    words = word_tokenize(review)
    documents.append((words, 'neg'))


random.shuffle(documents)

all_words = []

short_pos_words = word_tokenize(short_pos)
short_neg_words = word_tokenize(short_neg)

for word in short_pos_words:
    all_words.append(word.lower())

for word in short_neg_words:
    all_words.append(word.lower())

word_freq = nltk.FreqDist(all_words)

# word_features = list(word_freq.keys())[:3000]
word_features = [word for (word, count) in word_freq.most_common(4000)]


def find_features(document):
    words = set(document)
    features = {}
    for word in word_features:
        features[word] = (word in words)

    return features


feature_sets = [(find_features(rev), category)
                for (rev, category) in documents]


training_set = feature_sets[10:10000]
testing_set = feature_sets[10000:]


# Classifiers
classifier = nltk.NaiveBayesClassifier.train(training_set)

print('Most Distinctive Keywords: ')
classifier.show_most_informative_features(15)

print('Original Naive Bayes Classifier Accuracy: ',
      (nltk.classify.accuracy(classifier, testing_set)) * 100)


multinomial_classifier = SklearnClassifier(MultinomialNB())
multinomial_classifier.train(training_set)
print('Multinomial Naive Bayes Classifier Accuracy: ',
      (nltk.classify.accuracy(multinomial_classifier, testing_set)) * 100)


# gaussian_classifier = SklearnClassifier(GaussianNB())
# gaussian_classifier.train(training_set)
# print('Gaussian Naive Bayes Classifier Accuracy: ',
#       (nltk.classify.accuracy(gaussian_classifier, testing_set)) * 100)


bernoulli_classifier = SklearnClassifier(BernoulliNB())
bernoulli_classifier.train(training_set)
print('Bernoulli Naive Bayes Classifier Accuracy: ',
      (nltk.classify.accuracy(bernoulli_classifier, testing_set)) * 100)


logistic_classifier = SklearnClassifier(LogisticRegression())
logistic_classifier.train(training_set)
print('Logistic Regression Naive Bayes Classifier Accuracy: ',
      (nltk.classify.accuracy(logistic_classifier, testing_set)) * 100)


sgd_classifier = SklearnClassifier(SGDClassifier())
sgd_classifier.train(training_set)
print('SGD Naive Bayes Classifier Accuracy: ',
      (nltk.classify.accuracy(sgd_classifier, testing_set)) * 100)


svc_classifier = SklearnClassifier(SVC())
svc_classifier.train(training_set)
print('SVC Naive Bayes Classifier Accuracy: ',
      (nltk.classify.accuracy(svc_classifier, testing_set)) * 100)


linear_classifier = SklearnClassifier(LinearSVC())
linear_classifier.train(training_set)
print('Linear SVC Naive Bayes Classifier Accuracy: ',
      (nltk.classify.accuracy(linear_classifier, testing_set)) * 100)


nusvc_classifier = SklearnClassifier(NuSVC())
nusvc_classifier.train(training_set)
print('NuSVC Naive Bayes Classifier Accuracy: ',
      (nltk.classify.accuracy(nusvc_classifier, testing_set)) * 100)


# Voting with Classifiers

voted_classifier = VoteClassifier(classifier, multinomial_classifier, bernoulli_classifier,
                                  logistic_classifier, sgd_classifier, svc_classifier, linear_classifier, nusvc_classifier)


print('Voted Classifier Accuracy: ',
      (nltk.classify.accuracy(voted_classifier, testing_set)) * 100)


for feature_set in feature_sets[:10]:
    print('{} || Classification: {} with Confidence: {} %'.format(feature_set[1], voted_classifier.classify(
        feature_set[0]), voted_classifier.confidence(feature_set[0]) * 100))
