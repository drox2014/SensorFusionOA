import tensorflow as tf
import pandas as pd
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tokenize import word_tokenize
import numpy as np


class TextClassificationEngine:
    def __init__(self):
        self.__max_words = 50000
        # Max number of words in each complaint.
        self.__max_seq_length = 250
        # Stop words
        self.__stopwords_list = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
                                 "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
                                 "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
                                 "which", "who", "whom", "these", "those", "am", "is", "are", "was", "were", "be",
                                 "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a",
                                 "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at",
                                 "by", "for", "with", "against", "into", "through", "during", "before", "after",
                                 "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over",
                                 "under", "again", "further", "then", "once", "here", "there", "when", "why", "how",
                                 "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no",
                                 "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "don",
                                 "should", "now"]
        self.__labels = ['Locate', 'Describe', 'No_Op']
        self.__dataset_path = "/home/darshanakg/Projects/SensorFusion/zamia/data/dataset.txt"
        self.__tokenizer = self.__init_tokenizer()
        # Initializing the model
        config = tf.ConfigProto(intra_op_parallelism_threads=4,
                                inter_op_parallelism_threads=4,
                                allow_soft_placement=True,
                                device_count={'CPU': 2, 'GPU': 0})
        session = tf.Session(config=config)
        tf.keras.backend.set_session(session)
        self.__model = tf.keras.models.load_model("/home/darshanakg/Projects/SensorFusion/zamia/lstm.h5")

    def __init_tokenizer(self):
        df = pd.read_csv(self.__dataset_path, names=['sentence', 'operation'], sep=',', engine='python')
        sentences = df['sentence'].values
        filtered_sentences = self.filter_stopwords(sentences)
        detokenized_sentences = self.detokenize(filtered_sentences)
        tokenizer = tf.keras.preprocessing.text.Tokenizer(self.__max_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~',
                                                          lower=True)
        tokenizer.fit_on_texts(detokenized_sentences)
        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))
        return tokenizer

    def filter_stopwords(self, sentences):
        stopwords_set = set(self.__stopwords_list)
        filtered = []
        for sentence in sentences:
            tokenized_sentence = word_tokenize(sentence)
            filtered_sentence = []
            for w in tokenized_sentence:
                if w not in stopwords_set:
                    filtered_sentence.append(w)
            filtered.append(filtered_sentence)
        return filtered

    def detokenize(self, filtered_sentences):
        detokenized_sentences = []
        for sentence in filtered_sentences:
            detokenized_sentences.append(TreebankWordDetokenizer().detokenize(sentence))
        return detokenized_sentences

    def get_sentiment(self, command):
        new_command = [command]
        filtered_commands = self.filter_stopwords(new_command)
        seq = self.__tokenizer.texts_to_sequences(filtered_commands)
        padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=self.__max_seq_length)
        pred = self.__model.predict(padded)
        return self.__labels[np.argmax(pred)]


if __name__ == "__main__":
    e = TextClassificationEngine()
    new_command = 'What is this pen'
    print("Predicted Class: ", e.get_sentiment(new_command))
