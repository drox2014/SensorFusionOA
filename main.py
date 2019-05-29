from __future__ import print_function

import _thread

import numpy as np

from camera import CameraFeed
from operations import find_op

import os
from pocketsphinx import LiveSpeech, get_model_path

model_path = get_model_path()

speech = LiveSpeech(
    sampling_rate=16000,
    hmm=os.path.join(model_path, 'en-us'),
    lm='/home/darshanakg/Projects/SensorFusionOA/data/7723.lm.bin',
    dic='/home/darshanakg/Projects/SensorFusionOA/data/7723.dic'
)

words_file = open("data/words.txt", "r")
command_classes = {'count': 0, 'color': 1, 'focus': 2, 'no_op': 3}
command_classes_arr = ['count', 'color', 'focus', 'no_op']
object_classes = ['laptop', 'phone', 'book', 'bottle', 'pen', 'cup']
words = words_file.read().split('\n')[:-1]


# Calculate a unique decimal value for each sentence
def map_phrase(phrase):
    # Replace plural words with singular
    phrase = phrase.replace("books", "book")
    phrase = phrase.replace("laptops", "laptop")
    phrase = phrase.replace("phones", "phone")
    phrase = phrase.replace("cups", "cup")
    phrase = phrase.replace("bottles", "bottle")
    phrase = phrase.replace("pens", "pen")

    phrase_words = phrase.split()
    array = []
    object_id = -2
    for i in words:
        array.append((1, 0)[phrase_words.count(i) == 0])
        if phrase_words.count(i) and object_classes.count(i):
            object_id = object_classes.index(i) + 1
    return np.array(array), object_id


cf = CameraFeed(2)


def speech_event(camera_feed):
    for phrase in speech:
        print("SPEECH RECOG : ", phrase)
        input_vector, object_id = map_phrase(str(phrase).lower())
        opr = find_op(input_vector)
        camera_feed.process_frame(opr, object_id)


_thread.start_new_thread(speech_event, (cf,))
cf.start_camera()
