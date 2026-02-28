import sys
import jarvis

sys.argv = [
    "jarvis.py",
    "--input",
    "voice",
    "--output",
    "voice",
    "--push-to-talk",
    "on",
    "--interface",
    "ui",
]

jarvis.main()
