from statistics import mode
from nltk.classify import ClassifierI


class VoteClassifier(ClassifierI):

    def __init__(self, classifiers):
        self._classifiers = []
        for classifier in classifiers:
            self._classifiers.append(classifier)

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
