import concurrent.futures
import ssl
import requests
from urllib.error import HTTPError
import socket
import time 
import pandas as pd

ssl._create_default_https_context = ssl._create_unverified_context
MAX_THREADS = 30

def download_url(url, i):
    resp = requests.get(url)
    path = "C:/Users/Sandro/Documents/Python/M2/projet_fevrier/" + str(i) + '_HTML.txt'
    
    with open(path, "wb") as fh:
        fh.write(resp.content)
        
    time.sleep(0.25)
    return resp.content
    
def download_stories(story_urls, list_index):
    threads = min(MAX_THREADS, len(story_urls))
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        executor.map(download_url, story_urls, list_index)

def main(story_urls):
    t0 = time.time()
    download_stories(story_urls, list(range(len(story_urls))))
    t1 = time.time()
    print(f"{t1-t0} seconds to download {len(story_urls)} stories.")
    
main(LIST_URL)
