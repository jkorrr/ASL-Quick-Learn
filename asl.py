import undetected_chromedriver as uc
from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.chrome.options import *


adjusted_alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y']

if __name__ == '__main__':
    options = webdriver.ChromeOptions() 
    options.add_argument("start-maximized")
    driver = uc.Chrome(options = options)


    for letter in adjusted_alphabet:
        driver.get('https://www.signingsavvy.com/browse/' + letter)
        html_text = driver.page_source
        soup = BeautifulSoup(html_text, 'lxml')
        #spec_div = soup.find_all('span', _class = ;')
        
        image = soup.findAll('img')
        print(image)



