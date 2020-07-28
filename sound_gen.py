import os

import gtts

# commands = [
#     "find the laptop",
#     "show me the location of laptop",
#     "spot the laptop",
#     "locate the laptop",
#     "where is laptop",
#     "what is the location of my laptop",
#     "find the bottle",
#     "show me the location of bottle",
#     "spot the bottle",
#     "locate the bottle",
#     "where is bottle",
#     "what is the location of bottle",
#     "find the mouse",
#     "show me the location of mouse",
#     "spot the mouse",
#     "locate the mouse",
#     "where is mouse",
#     "what is the location of mouse",
#     "find the monitor",
#     "show me the location of monitor",
#     "spot the monitor",
#     "locate the monitor",
#     "where is monitor",
#     "what is the location of monitor",
#     "find the keyboard",
#     "show me the location of keyboard",
#     "spot the keyboard",
#     "locate the keyboard",
#     "where is keyboard",
#     "what is the location of keyboard"
# ]

commands = [
    "can I know more about",
    "describe",
    "give me details of",
    "i want details of",
    "provide details of",
    "what are details of",
]

objects = ["monitor", "keyboard", "laptop", "mouse", "bottle"]

for cmd in commands:
    for obj in objects:
        command = "%s %s" % (cmd, obj)
        tts = gtts.gTTS(command)
        tts.save(os.path.join("/home/darshanakg/speech_commands/new_describe", command.replace(" ", "_") + ".mp3"))
        print(os.path.join("/home/darshanakg/speech_commands/new_describe", command.replace(" ", "_") + ".mp3"))

