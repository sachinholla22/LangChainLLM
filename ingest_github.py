import requests
from bs4 import BeautifulSoup

def scrape_github_trending():
    url="https://github.com/trending"
    headers={'User-Agent':'Mozilla/5.0'}
    res=requests.get(url,headers=headers)
    soup=BeautifulSoup(res.text,"html.parser")

    results= []
    for repo in soup.select("article.Box-row"):
        title=repo.h2.text.strip().replace("\n","").replace(" ", "")
        description=repo.p.text.strip() if repo.p else "No Description"
        results.append({
            "title":title,
            "description":description
        })
    
    return results    

scrape_github_trending()