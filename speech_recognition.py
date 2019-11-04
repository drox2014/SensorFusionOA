from zamia.decode_mic import *
import os


class SpeechEngine:
    def __init__(self):
        self.sr = SpeechRecognizer()
        self.asr = self.sr.init_asr_kaldi()

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
                    r = self.sr.recognize_speech(self.asr)
                    s = te.get_sentiment(r)
                    print("[Speech] Detected speech: %s [%s]" % (r, s))
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
