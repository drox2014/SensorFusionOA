from datetime import datetime
from time import time


class LogRecord:
    def __init__(self):
        self.__start_time = datetime.now().timestamp() * 1000

    def save(self, tag, message):
        end_time = datetime.now().timestamp() * 1000
        elapsed_time = end_time - self.__start_time
        return "%d,%d,%d,%s,%s\n" % (self.__start_time,
                                     end_time,
                                     elapsed_time,
                                     tag,
                                     message)

    def get_start_time(self):
        return self.__start_time


class Logger:
    def __init__(self, tag):
        self.__log_file = open("./logs/%d-%s.txt" % (time(), tag), "w")
        self.__tag = tag
        self.__log_record = None
        self.__to_write = []

    def start(self):
        self.__log_record = LogRecord()
        return self.__log_record.get_start_time()

    def checkpoint(self, message):
        self.__to_write.append(self.__log_record.save(self.__tag, message))

    def add_flog(self, flag):
        timestamp = datetime.now().timestamp() * 1000
        self.__to_write.append("%d,%s\n" % (timestamp, flag))

    def save(self):
        for i in self.__to_write:
            self.__log_file.write(i)

        self.__to_write = []
        self.__log_file.flush()

    def close(self):
        self.__log_file.close()
