from analyzer import Analyzer


def init_analyzer():
    nlp_analyzer = Analyzer()
    nlp_analyzer.prepare_documents()
    nlp_analyzer.process_words()
    nlp_analyzer.init_classifiers()
    nlp_analyzer.init_voted_classifier(10)


init_analyzer()
