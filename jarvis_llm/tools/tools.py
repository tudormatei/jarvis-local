from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def get_user_info(user_id, special="none"):
    return f"User {user_id} details retrieved (special: {special})"


def get_weather_report(city, date):
    return f"Now searching weather for city: {city} at date {date}"


def play_song(song_name):
    brave_path = r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe"

    options = Options()
    options.binary_location = brave_path
    options.add_argument("--start-maximized")
    options.add_argument(
        "--user-data-dir=C:/Users/tudor/AppData/Local/BraveSoftware/Brave-Browser/User Data"
    )
    options.add_argument("--profile-directory=Default")
    options.add_experimental_option("detach", True)
    options.add_argument("--disable-logging")

    driver = webdriver.Chrome(options=options)

    driver.execute_script("window.open('');")

    driver.switch_to.window(driver.window_handles[-1])

    search_query = song_name.replace(" ", "+")
    search_url = f"https://www.youtube.com/results?search_query={search_query}"
    driver.get(search_url)

    wait = WebDriverWait(driver, 15)

    wait.until(EC.visibility_of_element_located((By.ID, "contents")))

    first_video = wait.until(
        EC.element_to_be_clickable(
            (By.CSS_SELECTOR, "#contents ytd-video-renderer #video-title")
        )
    )
    first_video.click()

    return f"Playing your request, sir."


if __name__ == "__main__":
    play_song("Sicko Mode")
