
import json
import time
import re

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager


def scrape_with_selenium():
    urls = [
        "https://www.isu.ac.bd/faculties/all-departments",
        "https://www.isu.ac.bd/faculties/department-of-computer-science-and-engineering",
        "https://isu.ac.bd/faculties/department-of-ai-and-data-science",
        "https://admission.isu.ac.bd/faculty-members",
        "https://isu.ac.bd/",
    ]

    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )

    all_scraped_data = []
    seen = set()

    def save_record(category, source_url, content):
        content = content.strip()

        if not content:
            return

        key = (source_url, content.lower())

        if key in seen:
            return

        seen.add(key)

        all_scraped_data.append({
            "category": category,
            "source_url": source_url,
            "content": content
        })

    try:
        for url in urls:
            try:
                print(f"\nScraping: {url}")
                driver.get(url)

                WebDriverWait(driver, 20).until(
                    lambda d: d.find_element(By.TAG_NAME, "body")
                )

                WebDriverWait(driver, 20).until(
                    lambda d: d.execute_script(
                        "return document.readyState"
                    ) == "complete"
                )

                time.sleep(3)

                # Trigger lazy-loaded content.
                previous_height = 0

                for _ in range(8):
                    height = driver.execute_script(
                        "return document.body.scrollHeight"
                    )

                    if height == previous_height:
                        break

                    previous_height = height

                    driver.execute_script(
                        "window.scrollTo(0, document.body.scrollHeight)"
                    )
                    time.sleep(1)

                driver.execute_script("window.scrollTo(0, 0)")
                time.sleep(1)

                print("Page title:", driver.title)
                print("Page URL:", driver.current_url)

                # Find faculty sections/cards where available.
                faculty_elements = driver.find_elements(
                    By.XPATH,
                    "//*[contains("
                    "translate(normalize-space(.),"
                    "'ABCDEFGHIJKLMNOPQRSTUVWXYZ',"
                    "'abcdefghijklmnopqrstuvwxyz'),"
                    "'faculty members')]"
                )

                # Keep the full rendered page text as fallback data.
                page_text = driver.execute_script(
                    "return document.body.innerText"
                )

                lines = [
                    " ".join(line.split())
                    for line in page_text.splitlines()
                    if line.strip()
                ]

                # Store ordinary page text in small chunks.
                chunk_size = 8

                for i in range(0, len(lines), chunk_size):
                    content = "\n".join(
                        lines[i:i + chunk_size]
                    )

                    if len(content.strip()) >= 3:
                        save_record(
                            "scraped_data",
                            url,
                            content
                        )

                # Also save individual visible faculty cards/sections.
                # This is a best-effort extraction: page layouts differ.
                saved_faculty = 0

                for element in faculty_elements:
                    try:
                        text = element.text.strip()

                        if not text:
                            continue

                        # Skip very large page containers.
                        if len(text) > 2500:
                            continue

                        # Avoid saving only a heading.
                        if len(text.splitlines()) < 2:
                            continue

                        save_record(
                            "faculty_section",
                            url,
                            text
                        )
                        saved_faculty += 1

                    except Exception:
                        continue

                print("Page text length:", len(page_text))
                print("Faculty sections collected:", saved_faculty)

                if "computer-science-and-engineering" in url:
                    print("\n--- CSE PAGE TEXT PREVIEW ---")
                    print(page_text[:5000])
                    print("--- END PREVIEW ---\n")

            except Exception as error:
                print(f"Error scraping {url}: {error}")

    finally:
        driver.quit()

    print("\nTotal records:", len(all_scraped_data))

    if not all_scraped_data:
        print("No data collected. Check URLs and browser errors.")
        return

    with open("isu_data.json", "w", encoding="utf-8") as f:
        json.dump(
            all_scraped_data,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("Data saved to isu_data.json")
    print("Inspect the output before rebuilding ChromaDB.")


if __name__ == "__main__":
    scrape_with_selenium()
