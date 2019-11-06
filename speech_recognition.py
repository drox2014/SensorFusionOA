from zamia.decode_mic import *
from multiprocessing import Queue
import os


class SpeechEngine:
    def __init__(self, queue: Queue):
        self.sr = SpeechRecognizer()
        self.asr = self.sr.init_asr_kaldi()
        self.__queue = queue

    def start_recognition(self):
        from text_classification import TextClassificationEngine
        te = TextClassificationEngine()
        p, stream = open_audio_stream()
        print("[Speech] Listening...")

        audio2send = []
        slid_win = deque(maxlen=int(SILENCE_LIMIT * REL) + 1)
        prev_audio = deque(maxlen=int(PREV_AUDIO * REL) + 1)
        started = False

        while True:
            try:
                cur_data = stream.read(CHUNK)
                slid_win.append(math.sqrt(abs(audioop.avg(cur_data, 4))))
                if sum([x > THRESHOLD for x in slid_win]) > 0:
                    if not started:
                        started = True
                    audio2send.append(cur_data)
                elif started is True:
                    # The limit was reached, finish capture and deliver.
                    filename = self.sr.save_speech(list(prev_audio) + audio2send, p)
                    text = self.sr.recognize_speech(self.asr)
                    sentiment = te.get_sentiment(text)
                    self.__queue.put(sentiment)
                    print("[Speech] Detected speech: %s [%s]" % (text, sentiment["operation"]))
                    # Remove temp file. Comment line to review.
                    os.remove(filename)
                    # Reset all
                    started = False
                    slid_win = deque(maxlen=int(SILENCE_LIMIT * REL) + 1)
                    prev_audio = deque(maxlen=int(PREV_AUDIO * REL) + 1)
                    audio2send = []
                else:
                    prev_audio.append(cur_data)
            except KeyboardInterrupt:
                break

        stream.stop_stream()
        stream.close()
        p.terminate()


if __name__ == "__main__":
    se = SpeechEngine()
    se.start_recognition()
