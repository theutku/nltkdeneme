import random
import nltk
from nltk.tokenize import word_tokenize


class DocumentProcessorBase:

    def __init__(self):
        self.documents = []
        self.all_words = []

    def open_files(self, samples):
        all_files = []

        for sample in samples:
            sample_file = open(sample, 'rb')
            decoded_sample_file = sample_file.read().decode('utf8', 'ignore')
            sample_file.close()
            all_files.append(decoded_sample_file)

        return all_files

    def process_files(self, files):
        # allowed = ['J']
        # for document in files:
        #     for sentence in document['file'].split('\n'):
        #         self.documents.append((sentence, document['category']))
        #         words = word_tokenize(sentence)
        #         pos = nltk.pos_tag(words)
        #         for word in pos:
        #             if word[1][0] in allowed:
        #                 self.all_words.append(word[0].lower())

        # random.shuffle(self.documents)
        for document in files:
            for sentence in document['file'].split('\n'):
                self.documents.append((sentence, document['category']))
                words = word_tokenize(sentence)
                for word in words:
                    self.all_words.append(word.lower())

        random.shuffle(self.documents)
