# from nltk.tokenize import sent_tokenize, word_tokenize
# from nltk.corpus import stopwords

# stops = set(stopwords.words('english'))

# phrase = "Uncertainty about who will lead Brexit divorce talks for
# Britain is a very real problem, the diplomat who helped draft article 50
# has said, as he warned the UK faces a 45% chance of crashing out of the
# EU with no deal."

# words = word_tokenize(phrase)

# for word in sent_tokenize(phrase):
#     print(word)

# filtered = []

# for word in words:
#     if word not in stops:
#         filtered.append(word)

# print('Filtered Sentence: \n', filtered)

# filtered_stops = [word for word in words if word in stops]
# print('Filtered Stops: \n', filtered_stops)

# import nltk
# from nltk.tokenize import PunktSentenceTokenizer
# from nltk.corpus import state_union


# train = state_union.raw('2005-GWBush.txt')
# sample = state_union.raw('2006-GWBush.txt')

# custom_tokenizer = PunktSentenceTokenizer(train)

# tokenized = custom_tokenizer.tokenize(sample)


# def process_tokenized():
#     try:
#         for sentence in tokenized[:10]:
#             words = nltk.word_tokenize(sentence)
#             tags = nltk.pos_tag(words)

#             # # Chunking
#             # chunk_gram = """Chunk: {<RB.?>?<VB.?>?<NNP>+<NN>?} """
#             # chunk_parser = nltk.RegexpParser(chunk_gram)
#             # chunked = chunk_parser.parse(tags)
#             # chunked.draw()

#             # # Named Entities
#             # named_entities = nltk.ne_chunk(tags)
#             # named_entities.draw()
#     except Exception as e:
#         print(str(e))


# process_tokenized()

import random
import nltk
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode


class VoteClassifier(ClassifierI):

    def classify(self, features):
        votes = []
        for classifier in self._classifiers:
            vote = classifier.classify(features)
            votes.append(vote)

        return mode(votes)

    def confidence(self, features):
        votes = []
        for classifier in self._classifiers:
            vote = classifier.classify(features)
            votes.append(vote)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

    def __init__(self, *classifiers):
        self._classifiers = classifiers


documents = []

for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        documents.append((list(movie_reviews.words(fileid)), category))

random.shuffle(documents)
# print(documents[1])

all_words = []

for word in movie_reviews.words():
    all_words.append(word.lower())

word_freq = nltk.FreqDist(all_words)

# print(word_freq.most_common(15))
# print(word_freq['exciting'])

word_features = list(word_freq.keys())[:3000]


def find_features(document):
    words = set(document)
    features = {}
    for word in word_features:
        features[word] = (word in words)

    return features


feature_sets = [(find_features(rev), category)
                for (rev, category) in documents]


training_set = feature_sets[:1900]
testing_set = feature_sets[1900:]


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
      (nltk.classify.accuracy(voted_classifier, testing_set) * 100))


print('Classification: ', (voted_classifier.classify(
    testing_set[0][0]) * 100))

print('Confidence: ', (voted_classifier.confidence(testing_set[0][0]) * 100))
