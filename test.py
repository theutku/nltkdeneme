from analyzer import Analyzer
from twitterbase import TwitterListener
from tweepy import Stream


def init_twitter_analyzer(subject):
    # nlp_analyzer = Analyzer()
    # nlp_analyzer.prepare_documents()
    # nlp_analyzer.process_words()
    # nlp_analyzer.init_classifiers()
    # nlp_analyzer.init_voted_classifier(10)

    twitter_listener = TwitterListener()
    twitter_listener.init_listener()
    twitterStream = Stream(twitter_listener.auth, twitter_listener)
    twitterStream.filter(track=[subject])


init_twitter_analyzer('movie')
