# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 21:39:26 2020

@author: Dave
"""

from bs4 import BeautifulSoup
from urllib.request import urlopen

def parsePrice():
    url = 'https://finance.yahoo.com/quote/AAPL?p=AAPL&.tsrc=fin-srch'
    try:
        page = urlopen(url)
    except:
        print("Error opening the URL")
    page = urlopen(url)
    soup = BeautifulSoup(page, "html.parser")
    price = soup.find('div',{'class': 'My(6px) Pos(r) smartphone_Mt(6px)'}).find('span').text
    return price

while True:
    print("The current price is: "+str(parsePrice()))