# import requests


# def web_scrape(url: str) -> str:
#     """Fetch and return the text content of a web page."""
#     response = requests.get(url)
#     response.raise_for_status()
#     soup = BeautifulSoup(response.text, "html.parser")
#     return soup.get_text()
