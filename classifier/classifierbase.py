import nltk
from nltk.classify.scikitlearn import SklearnClassifier

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from classifier.votedclassifier import VoteClassifier


class ClassifierBase:

    def __init__(self):
        self._classifiers = []
        self.sklearn_classifiers = []

    def init_default_classifier(self, training_set, testing_set):
        classifier = {'name': 'Original Naive Bayes Classifier',
                      'classifier': nltk.NaiveBayesClassifier.train(training_set)}
        print('{} Accuracy: {} %'.format(
            classifier['name'], (nltk.classify.accuracy(classifier['classifier'], testing_set)) * 100))
        self._classifiers.append(classifier)
        return classifier

    def init_sklearn_classifiers(self):
        multinomial_classifier = {'name': 'Multinomial Naive Bayes Classifier',
                                  'classifier': SklearnClassifier(MultinomialNB())}
        self._classifiers.append(multinomial_classifier)

        bernoulli_classifier = {'name': 'Bernoulli Naive Bayes Classifier',
                                'classifier': SklearnClassifier(BernoulliNB())}
        self._classifiers.append(bernoulli_classifier)

        logistic_classifier = {'name': 'Logistic Regression Naive Bayes Classifier',
                               'classifier': SklearnClassifier(LogisticRegression())}
        self._classifiers.append(logistic_classifier)

        sgd_classifier = {'name': 'SGD Naive Bayes Classifier',
                          'classifier': SklearnClassifier(SGDClassifier())}
        self._classifiers.append(sgd_classifier)

        # svc_classifier = {'name': 'SVC Naive Bayes Classifier',
        #                   'classifier':  SklearnClassifier(SVC())}
        # self._classifiers.append(svc_classifier)

        linear_classifier = {'name': 'Linear SVC Naive Bayes Classifier',
                             'classifier': SklearnClassifier(LinearSVC())}
        self._classifiers.append(linear_classifier)

        nusvc_classifier = {'name': 'NuSVC Naive Bayes Classifier',
                            'classifier': SklearnClassifier(NuSVC())}
        self._classifiers.append(nusvc_classifier)

        for c in self._classifiers:
            if c['name'] != 'Original Naive Bayes Classifier':
                self.sklearn_classifiers.append(c)

    def train_classifiers(self, training_set, testing_set):
        for classifier in self.sklearn_classifiers:
            classifier['classifier'].train(training_set)
            print('{} Accuracy: {} %'.format(
                classifier['name'], (nltk.classify.accuracy(classifier['classifier'], testing_set)) * 100))

    def init_voted_classifier(self, features, testing_set):
        voted_classifier = {'name': 'Voted Classifier',
                            'classifier': VoteClassifier(self._classifiers)}

        self.voted_classifier = voted_classifier

        print('{} Accuracy: {} %'.format(
            voted_classifier['name'], (nltk.classify.accuracy(voted_classifier['classifier'], testing_set)) * 100))

        for feature_set in features:
            conf = voted_classifier['classifier'].confidence(feature_set[0])
            if type(conf) != str:
                conf = conf * 100
            print('{} || Voted Classification: {} with Confidence: {} %'.format(feature_set[1], voted_classifier['classifier'].classify(
                feature_set[0]), conf))
