import shutil
from multiprocessing import Queue
from time import sleep

import psutil

from onlineasr.online_speech_recognition import OnlineSpeechRecognizer
from utils.logger import Logger
from zamia.decode_mic import *


class SpeechEngine:
    def __init__(self, queue: Queue):
        from config import config
        process = psutil.Process(os.getpid())
        start = process.memory_info()[0]
        if config.ASR == 1:
            self.sr = SpeechRecognizer()
        else:
            self.sr = OnlineSpeechRecognizer()
        usage = process.memory_info()[0] - start
        print("[Memory Usage | Speech Recognition]", usage >> 20)
        self.__queue = queue
        self.__logger_speech = Logger("speech")
        self.__logger_text = Logger("text")

    def start_recognition(self):
        from text_classification import TextClassificationEngine
        from config import config
        te = TextClassificationEngine()
        if config.TC == 1:
            get_sentiment = te.get_sentiment
        else:
            get_sentiment = te.get_svm_sentiment

        directory = "/home/darshanakg/speech_commands/new_describe"
        for file_name in os.listdir(directory):
            source = os.path.join(directory, file_name)
            destination = "/home/darshanakg/Projects/SensorFusion/zamia/aspire_new/data/test/utt1.wav"
            if os.path.exists(destination):
                os.remove(destination)
            shutil.copyfile(source, destination)
            self.__logger_speech.start()
            text = self.sr.recognize_speech()
            text = text.strip().lower()
            self.__logger_speech.checkpoint("%s,%s" % (file_name, text))
            self.__logger_text.start()
            sentiment = get_sentiment(text)
            if sentiment:
                self.__queue.put(sentiment)
                self.__logger_text.checkpoint("%s,%s,%s" % (file_name, text, sentiment["operation"]))
                print("[Speech] Detected speech: %s [%s]" % (text, sentiment["operation"]))
            else:
                print("[Speech] Detected speech: %s [Invalid Command]" % text)
            sleep(10)
        self.__logger_speech.save()
        self.__logger_speech.close()
        self.__logger_text.save()
        self.__logger_text.close()
        print("IT'S OVER")


if __name__ == "__main__":
    queue = Queue()
    se = SpeechEngine(queue)
    se.start_recognition()
