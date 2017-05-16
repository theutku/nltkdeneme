from analyzer import Analyzer
from twitterbase import TwitterListener
from tweepy import Stream


def init_twitter_analyzer(subject):

    twitter_listener = TwitterListener()
    twitter_listener.init_analyzer(10)
    twitter_listener.init_listener()
    twitterStream = Stream(twitter_listener.auth, twitter_listener)
    twitterStream.filter(track=[subject])


if __name__ == '__main__':
    init_twitter_analyzer('Donald Trump')
