import webview
import logging

logger = logging.getLogger(__name__)


class Api:
    """Api that is called from inside JS when audio is finished playing"""

    def audioFinished(self):
        global voice_finished
        voice_finished = True

    def muteUnmute(self):
        global muted
        muted = not muted


def print_message(isUser, message):
    global window

    escaped_message = message.replace("'", "\\'").replace('"', '\\"')

    js_boolean = 'true' if isUser else 'false'

    window.evaluate_js(f"displayLine({js_boolean}, '{escaped_message}')")


def showLoader(show):
    global window

    js_boolean = 'true' if show else 'false'

    window.evaluate_js(f"displayLoader({js_boolean})")


def cleanup():
    logger.info("UI Closing window.")


def main():
    global window
    js_api = Api()

    width = 400
    height = 700

    window = webview.create_window(
        'J.A.R.V.I.S.', './ui/index.html', js_api=js_api, width=width, height=height, frameless=True, easy_drag=True)
    logger.info("UI Started window.")
    window.events.closed += cleanup

    webview.start(debug=True)


if __name__ == '__main__':
    main()
