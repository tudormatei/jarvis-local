import webview
import logging

logger = logging.getLogger(__name__)


class Api:
    """Api that is called from inside JS when audio is finished playing"""

    def __init__(self):
        pass


class JarvisUI:
    def __init__(self, width=400, height=700, html_path="./ui/index.html"):
        self.html_path = html_path
        self.width = width
        self.height = height
        self.window = None
        self.voice_finished = False
        self.muted = False

        self.api = Api()

    def print_message(self, isUser, message):
        escaped_message = message.replace("'", "\\'").replace('"', '\\"')
        js_boolean = 'true' if isUser else 'false'
        self.window.evaluate_js(
            f"displayLine({js_boolean}, '{escaped_message}')")

    def showLoader(self, show):
        js_boolean = 'true' if show else 'false'
        self.window.evaluate_js(f"displayLoader({js_boolean})")

    def cleanup(self):
        logger.info("UI Closing window.")

    def start(self):
        self.window = webview.create_window(
            'J.A.R.V.I.S.', url=self.html_path, js_api=self.api,
            width=self.width, height=self.height, frameless=True, easy_drag=True
        )
        logger.info("UI Started window.")
        self.window.events.closed += self.cleanup
        webview.start(debug=False)


if __name__ == '__main__':
    ui = JarvisUI()
    ui.start()
