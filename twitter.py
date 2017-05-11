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
            vote = c['classifier'].classify(features)
            votes.append(vote)

        try:
            md = mode(votes)
            return md
        except:
            return 'Classification Draw!'

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            vote = c['classifier'].classify(features)
            votes.append(vote)

        try:
            choice_votes = votes.count(mode(votes))
            confid = float(choice_votes) / len(votes)
            return confid
        except:
            return 'Confidence Draw!'


short_pos = open('short_reviews/positive.txt', 'rb').read()
short_neg = open('short_reviews/negative.txt', 'rb').read()

short_pos = short_pos.decode('utf8', 'ignore')
short_neg = short_neg.decode('utf8', 'ignore')

documents = []
all_words = []

allowed_word_types = ['J']

for review in short_pos.split('\n'):
    documents.append((review, 'pos'))
    words = word_tokenize(review)
    pos = nltk.pos_tag(words)
    for word in pos:
        if word[1][0] in allowed_word_types:
            all_words.append(word[0].lower())

for review in short_neg.split('\n'):
    documents.append((review, 'neg'))
    words = word_tokenize(review)
    pos = nltk.pos_tag(words)
    for word in pos:
        if word[1][0] in allowed_word_types:
            all_words.append(word[0].lower())


# short_pos.close()
# short_neg.close()

random.shuffle(documents)

word_freq = nltk.FreqDist(all_words)


word_features = [word for (word, count) in word_freq.most_common(4000)]


def find_features(document):
    words = word_tokenize(document)
    word_set = set(words)
    features = {}
    for word in word_features:
        features[word] = (word in word_set)

    return features


feature_sets = [(find_features(rev), category)
                for (rev, category) in documents]


training_set = feature_sets[10: 10000]
testing_set = feature_sets[10000:]


# Classifiers
classifier = {'name': 'Original Naive Bayes Classifier',
              'classifier': nltk.NaiveBayesClassifier.train(training_set)}

print('Most Distinctive Keywords: ')
classifier['classifier'].show_most_informative_features(15)


multinomial_classifier = {'name': 'Multinomial Naive Bayes Classifier',
                          'classifier': SklearnClassifier(MultinomialNB())}


bernoulli_classifier = {'name': 'Bernoulli Naive Bayes Classifier',
                        'classifier': SklearnClassifier(BernoulliNB())}


logistic_classifier = {'name': 'Logistic Regression Naive Bayes Classifier',
                       'classifier': SklearnClassifier(LogisticRegression())}


sgd_classifier = {'name': 'SGD Naive Bayes Classifier',
                  'classifier': SklearnClassifier(SGDClassifier())}


svc_classifier = {'name': 'SVC Naive Bayes Classifier',
                  'classifier':  SklearnClassifier(SVC())}


linear_classifier = {'name': 'Linear SVC Naive Bayes Classifier',
                     'classifier': SklearnClassifier(LinearSVC())}

nusvc_classifier = {'name': 'NuSVC Naive Bayes Classifier',
                    'classifier': SklearnClassifier(NuSVC())}


classifiers = [classifier, multinomial_classifier, bernoulli_classifier, logistic_classifier,
               sgd_classifier, svc_classifier, linear_classifier, nusvc_classifier]


for classifier in classifiers:
    classifier['classifier'].train(training_set)
    print('{} Accuracy: {} %'.format(
        classifier['name'], (nltk.classify.accuracy(classifier['classifier'], testing_set)) * 100))


# Voting with Classifiers

voted_classifier = {'name': 'Voted Classifier', 'classifier': VoteClassifier(classifier, multinomial_classifier, bernoulli_classifier,
                                                                             logistic_classifier, sgd_classifier, svc_classifier, linear_classifier, nusvc_classifier)}

for feature_set in feature_sets[:10]:
    print('{} || Voted Classification: {} with Confidence: {} %'.format(feature_set[1], voted_classifier['classifier'].classify(
        feature_set[0]), voted_classifier['classifier'].confidence(feature_set[0]) * 100))


def analyze_sentiment(text):
    feats = find_features(text)
    for classifier in classifiers:
        sentiment = classifier['classifier'].classify(feats)
        conf = classifier['classifier'].confidence(feats)
        print('Classification: {} with Confidence: {} %'.format(
            sentiment, conf * 100))
        return sentiment, conf
