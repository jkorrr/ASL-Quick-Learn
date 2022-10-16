from bs4 import BeautifulSoup
from selenium.webdriver.chrome.options import *
import requests
import os
from urllib.request import urlopen
import urllib.request
from PIL import Image
  


adjusted_alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y']

def scraper():
    """
    options = webdriver.ChromeOptions() 
    options.add_argument("start-maximized")
    driver = uc.Chrome(options = options)
    images = []
    """
    images = []

    for letter in adjusted_alphabet:
        html_text = urlopen('https://www.signingsavvy.com/browse/' + letter)
        #html_text = driver.page_source
        soup = BeautifulSoup(html_text, 'lxml')
        for item in soup.find_all('img'):
            if 'jpg' in (item['src']):
                images.append('https://www.signingsavvy.com/' + item['src'])
    print(images)
    return images



def folder_create(images):
    urllib.request.urlretrieve('https://www.signingsavvy.com/images/words/alphabet/2/v1.jpg', "gfg.png")
  
    img = Image.open("gfg.png")
    img.show()
  
    try:
        folder_name = "Images"
        # folder creation
        os.mkdir(folder_name)
 
    # if folder exists with that name, ask another name
    except:
        print("Folder Exist with that name!")
        folder_create()
 
    # image downloading start
    download_images(scraper(), folder_name)
 
 
# DOWNLOAD ALL IMAGES FROM THAT URL
def download_images(images, folder_name):
   
    # initial count is zero
    count = 0
 
    # print total images found in URL
 
    # checking if images is not zero
    if len(images) != 0:
        for image_link in images:
            
            try:
                urllib.request.urlretrieve(image_link, "jpg")
                img = Image.open("jpg")
                
                count += 1
            except:
                pass
 
        # There might be possible, that all
        # images not download
        # if all images download
        if count == len(images):
            print("All Images Downloaded!")
             
        # if all images not download
        else:
            print(f"Total {count} Images Downloaded Out of {len(images)}")


folder_name = "Images"
try:
    os.mkdir(folder_name)
except:
    pass
download_images(scraper(), folder_name)
