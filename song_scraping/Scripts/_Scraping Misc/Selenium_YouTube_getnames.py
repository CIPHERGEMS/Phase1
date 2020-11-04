# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 21:21:51 2020

@author: Dave
"""

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC

from pathlib import Path
import os

home = os.path.expanduser('~')
homebase_path = Path(home)
os.chdir(homebase_path / "Desktop" / "scraping" )
youtube_path = homebase_path / "Desktop" / "Scrape_YouTube" 
os.makedirs(youtube_path, exist_ok = True)
pickle_folder = youtube_path / "Pickle"
os.makedirs(pickle_folder, exist_ok = True)
 

options = webdriver.ChromeOptions() 
options.add_argument("start-maximized")
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)
driver = webdriver.Chrome(executable_path = str(homebase_path / "Desktop" / "scraping" / "chromedriver.exe"))
baseurl = "http://youtube.com"
keyword = 'Eminem'
driver.get(f'{baseurl}/search?q={keyword}')
print([my_elem.text for my_elem in WebDriverWait(driver, 20).until(EC.visibility_of_all_elements_located((By.XPATH, "//yt-formatted-string[@class='style-scope ytd-video-renderer' and @aria-label]")))])
driver.quit()

