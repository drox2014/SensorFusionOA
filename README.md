# SensorFusion
A sensor fusion framwork that fuses audio, visual, and gestural streams together

# Dependecies 
1. Install Tensorflow-gpu 1.13
2. Follow installation guide provided with this [link](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)
3. Install opencv python using `pip install opencv-python`
4. Setup pocket-sphinx as mentioned in this [link](https://github.com/uom-cse-realitix/speech_recognition)
5. Install pykaldi as mentioned in this [link](https://github.com/pykaldi/pykaldi#installation)

# Run Main file

```
source zamia/path.sh
LD_PRELOAD=./libLeap.so python main.py
```
