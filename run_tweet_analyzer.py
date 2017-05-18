from analyzer.analyzerbase import Analyzer
from twitter.twitterbase import TwitterListener
from tweepy import Stream
# import classifier.plotter as plotter


def init_twitter_analyzer(subject):

    twitter_listener = TwitterListener()

    try:
        pickled_instance = open('saved/analyzer.pickle')
        print('Saved instance found. Analyzer will not be re-instantiated.')
        twitter_listener.init_analyzer(10)

    except:
        print('Saved instance not found. Analyzer will be instantiated.')
        twitter_listener.init_analyzer(10, first_run=True)

    try:
        twitter_listener.init_listener()
        twitterStream = Stream(twitter_listener.auth, twitter_listener)
        twitterStream.filter(track=[subject])
    except Exception as e:
        print(e)


if __name__ == '__main__':
    init_twitter_analyzer('Donald Trump')
