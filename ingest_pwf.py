import requests
from bs4 import BeautifulSoup

def scrape_pwc_latest():
    url = "https://paperswithcode.com/latest"
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "html.parser")

    papers = []
    for paper in soup.select("div.paper-card"):
        title_tag = paper.select_one("h1 a")
        summary_tag = paper.select_one("p")

        if title_tag and summary_tag:
            title = title_tag.text.strip()
            summary = summary_tag.text.strip()
            papers.append({
                "title": title,
                "summary": summary
            })
    
    return papers

# Run the function for testing
scrape_pwc_latest()
