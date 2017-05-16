from analyzer import Analyzer
import json

from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener


class TwitterListener(StreamListener):

    def init_analyzer(self, example_count):
        self.analyzer = Analyzer()
        self.analyzer.init_analyzer(example_count)

    def init_listener(self):
        print('Initializing Twitter Stream...')
        consumer_key = 'SmsP3MaBZwEsoD7SeajQVfbj5'
        consumer_secret = 'rexAFZLxD0Z0zNVMr2dlnC283o0a6A97DGtndx38u5G68PS0Vm'
        access_token = '859723615880851456-1EnuyWtSPcoqvX8WZVn7EuaH6WHqfF6'
        access_token_secret = 'AiqYTDDnRJkoOdbNmPE3RP5MmzaO7Je1CZnnAFYu0DIds'

        self.auth = OAuthHandler(consumer_key, consumer_secret)
        self.auth.set_access_token(access_token, access_token_secret)

        # self.twitterStream = Stream(self.auth, self)
        # self.twitterStream.filter(track=['happy'])

    def on_data(self, data):
        try:
            all_data = json.loads(data)

            if len(all_data['text']) != 0:
                tweet = all_data["text"]
                sentiment_value, confidence = self.analyzer.analyze_tweet(
                    tweet)
                print(tweet, sentiment_value, confidence)

                if type(confidence) == float and confidence * 100 >= 50:
                    output = open("results/twitter-out.txt", "a")
                    output.write(
                        '{} --- {} | Confidence: {} %'.format(tweet, sentiment_value, confidence))
                    output.write('\n')
                    output.close()

            return True
        except Exception as e:
            print(e)
            return True

    def on_error(self, status):
        print(status)
