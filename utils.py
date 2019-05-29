import os
from pocketsphinx import LiveSpeech, get_model_path

model_path = get_model_path()

speech = LiveSpeech(
    sampling_rate=16000,
    hmm=os.path.join(model_path, 'en-us'),
    lm='/home/darshanakg/Projects/SensorFusionOA/data/7723.lm.bin',
    dic='/home/darshanakg/Projects/SensorFusionOA/data/7723.dic'
)


def speech_recognition():
    for phrase in speech:
        print(phrase)


speech_recognition()
