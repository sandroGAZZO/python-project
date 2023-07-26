## Importations
import concurrent.futures
#%%
import ssl
#%%
import requests
#%%
from urllib.error import HTTPError
#%%
import socket
#%%
import time
#%%
from bs4 import BeautifulSoup
#%%
import pandas as pd
#import texthero as hero
#%%
## Enregistrement en .txt du code de pages web à partir d'une liste d'URLs*

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


main(["https://en.wikipedia.org/wiki/Charles_Baudelaire"])

## Création d'un Data Frame comprenant le texte et sa balise associée à partir du .txt précédent
def remove_html_markup(s):
    tag = False
    quote = False
    out = ""

    for c in s:
            if c == '<' and not quote:
                tag = True
            elif c == '>' and not quote:
                tag = False
            elif (c == '"' or c == "'") and tag:
                quote = not quote
            elif not tag:
                out = out + c

    return out

# Liste des balises à extraire
list_bal = ['div','p','h1', 'h2' ,'title']

# Initialisation du Data Frame
df = pd.DataFrame(columns=['Text_original','Text_tag'])
index = 0

# Importation du .txt sous Python dans la forme adéquate pour l'extraction du texte
try:
    path = "C:/Users/Sandro/Documents/Python/M2/projet_fevrier/" + str(0) + '_HTML.txt'
    f = open(path,"r",encoding='utf-8')
    txt = f.read()
    soup = BeautifulSoup(txt,'html.parser')
except:
    print("FileNotFoundError")

f.close()

# Remplissage du Data Frame
for balise in list:
        for i in range(len(soup.findAll(balise))):
            index += 1
            newline = remove_html_markup(str(soup.findAll(balise)[i]))
            df.loc[index] = [newline,balise]

##

print(df)