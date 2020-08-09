from selenium import webdriver
from datetime import date
import pandas as pd
import re
from time import sleep
import os

newsQuaryfront = "https://search.naver.com/search.naver?query="
newsQuaryend = "&where=news&ie=utf8&sm=nws_hty"

newsTitlefront1 = "/html/body/div[3]/div[2]/div/div[1]/div[1]/ul/li["
newsTitlefront2 = "/html/body/div[3]/div[2]/div/div[1]/div/ul/li["
newsTitleend = "]/dl/dt/a"
newsTabfront = "/html/body/div[3]/div[2]/div/div[1]/div[1]/div[2]/a["
option = webdriver.ChromeOptions()
#option.add_argument("headless")
option.add_argument("window-size=1920x1080")
#option.add_argument("disable-gpu")
newsTab = [1,3,4,5,6,6,6,6,6,6]
hangul = re.compile('[^ 0-9ㄱ-ㅣ가-힣]+')

def getsource(myurl):
    import requests
    sources = []
    for url in myurl:
        try:
            html = requests.get(myurl).text
        except:
            continue
        try:
            only_hangul = hangul.sub('', html)
        except:
            print(html)
        only_meaningful = re.sub('[0-9]{5,}', ' ', only_hangul)
        sources.append(only_meaningful)
    return sources


def newcollactor(quary):
    if os.path.exists("./data/"+quary+str(date.today())+"data.csv"):
        data = pd.read_csv("./data/"+quary+str(date.today())+"data.csv")
        return data

    driver = webdriver.Chrome('./data/chrome83.14/chromedriver', options=option)
    driver.implicitly_wait(3)

    driver.get(newsQuaryfront + quary + newsQuaryend)
    #main_tab = driver.current_window_handle

    urllist = []
    for tab in newsTab:
        for news in range(1, 11):
            try:
                urllist.append(driver.find_element_by_xpath(newsTitlefront1 + str(news) + newsTitleend).get_attribute('href'))
            except:
                urllist.append(driver.find_element_by_xpath(newsTitlefront2 + str(news) + newsTitleend).get_attribute('href'))
            #driver.switch_to.window(driver.window_handles[1])
            #driver.switch_to.window(main_tab)
        driver.find_element_by_xpath(newsTabfront + str(tab) + "]").click()
    driver.quit()


    from multiprocessing import Pool
    myp = Pool(4)
    with myp:
        sources = myp.map(getsource, urllist)
    import numpy as np
    sources = np.array(sources).flatten()
    data = pd.DataFrame(sources, columns=["crawled"])
    sleep(0.1)
    data.to_csv("./data/"+quary+str(date.today())+"data.csv", encoding='utf-8')

    return data


if __name__ == "__main__":
    newcollactor("CJ제일제당")