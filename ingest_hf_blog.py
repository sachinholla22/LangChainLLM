import requests
from bs4 import BeautifulSoup
import json

def scrape_hf_blogs():
    url = "https://huggingface.co/blog"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    blogs = []
    for div in soup.find_all("div", class_="SVELTE_HYDRATER contents"):
        try:
            data_props = div.get("data-props")
            if data_props:
                blog_data = json.loads(data_props)["blog"]
                title = blog_data["title"]
                slug = blog_data["slug"]
                date = blog_data["publishedAt"][:10]
                author = blog_data["authorData"]["fullname"]
                url = "https://huggingface.co/blog/" + slug

                blogs.append({
                    "title": title,
                    "author": author,
                    "published": date,
                    "url": url
                })
        except Exception as e:
            print("Skipping one due to error:", e)

    
    return blogs

# Run this to test
scrape_hf_blogs()
