from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import SpeechToTextV1

from onlineasr.utils import Utils

config = Utils.readYaml("onlineasr/config.yaml")


class OnlineSpeechRecognizer:
    def __init__(self):
        self.__path = "zamia/aspire_new/data/test/utt1.wav"

        # initialize speech to text service
        self.__authenticator = IAMAuthenticator(config['watson']['API_KEY'])
        self.__service = SpeechToTextV1(authenticator=self.__authenticator)
        self.__service.set_service_url(config['watson']['URL'])

    def recognize_speech(self):
        audio_file = open(self.__path, 'rb')
        result = self.__service.recognize(audio=audio_file, content_type='audio/wav', timestamps=False,
                                          word_confidence=False).get_result()
        text = result['results'][result['result_index']]['alternatives'][0]['transcript']
        return text
