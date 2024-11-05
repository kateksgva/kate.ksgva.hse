import requests 
from bs4 import BeautifulSoup
url = 'https://ru.wikipedia.org/wiki/%D0%A5%D1%8D%D0%BB%D0%BB%D0%BE%D1%83%D0%B8%D0%BD'
r_wiki = requests.get(url=url)
soup_wiki = BeautifulSoup(r_wiki.text, 'html.parser')
pars_wiki = soup_wiki.find_all('p')
texts_wiki = [text.get_text() for text in pars_wiki]
texts_wiki
