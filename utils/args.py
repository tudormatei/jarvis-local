import argparse
from enum import Enum


class InputMode(Enum):
    TEXT = 'text'
    VOICE = 'voice'


class OutputMode(Enum):
    TEXT = 'text'
    VOICE = 'voice'


class OutputInterface(Enum):
    UI = 'ui'
    CLI = 'cli'


class PushToTalk(Enum):
    ON = 'on'
    OFF = 'off'


class LogLevel(Enum):
    INFO = 'info'
    NONE = 'none'


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run JARVIS with flexible input/output modes."
    )
    parser.add_argument('--input', type=str,
                        choices=[e.value for e in InputMode], required=True)
    parser.add_argument('--output', type=str,
                        choices=[e.value for e in OutputMode], required=True)
    parser.add_argument('--interface', type=str,
                        choices=[e.value for e in OutputInterface], required=True)
    parser.add_argument('--log', type=str,
                        choices=[e.value for e in LogLevel], default=LogLevel.INFO.value)
    parser.add_argument('--push-to-talk', type=str,
                        choices=[e.value for e in PushToTalk])
    args = parser.parse_args()

    if args.input == InputMode.VOICE.value and args.push_to_talk is None:
        parser.error("--push-to-talk is required when --input voice")
    if args.input == InputMode.TEXT.value and args.push_to_talk is not None:
        parser.error("--push-to-talk is only valid with --input voice")

    args.input = InputMode(args.input)
    args.output = OutputMode(args.output)
    args.interface = OutputInterface(args.interface)
    args.log = LogLevel(args.log)
    if args.push_to_talk is not None:
        args.push_to_talk = PushToTalk(args.push_to_talk)
    return args
