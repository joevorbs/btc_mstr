import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from datetime import datetime
import time


#Start two webdrivers for mstr and btc
driver1 = webdriver.Chrome("/Users/joevorbeck/desktop/chromedriver")
driver2 = webdriver.Chrome("/Users/joevorbeck/desktop/chromedriver")

#Open $btc on google
driver1.get("https://www.google.com/search?q=%24btc&oq=%24btc&aqs=chrome.0.69i59j0l7.2409j0j7&sourceid=chrome&ie=UTF-8")
#Open $mstr on google
driver2.get("https://www.google.com/search?q=%24mstr&oq=%24mstr&aqs=chrome.0.69i59l2j0i271l3j69i59.899j0j7&sourceid=chrome&ie=UTF-8")

#Blank DFs 
btc_df = pd.DataFrame(columns = ['price', 'timestamp'])
mstr_df = pd.DataFrame(columns = ['price', 'timestamp'])

#Scrape BTC and MSTR prices every minute for 100 days
for i in range(0, 14400):
    current_time = datetime.now().time()   #Get current time for each iteration 
    btc_price = driver1.find_element_by_xpath("//span[@data-value]").get_attribute('innerHTML')  #Scrape btc price
    mstr_price = driver2.find_element_by_xpath("//span[@jsname = 'vWLAgc']").get_attribute('innerHTML') #Scrape microstrategy price
    
    btc_list = [btc_price, current_time]      #Add current prices to a list
    mstr_list = [mstr_price, current_time]    #along with the current time
    
    #print(btc_list)
    #print(mstr_list)
    
    #append current price and time to blank DFs
    btc_df = btc_df.append({"price": btc_list[0], "timestamp": btc_list[1]}, ignore_index = True)
    mstr_df = mstr_df.append({"price": mstr_list[0], "timestamp": mstr_list[1]}, ignore_index = True) 
    
    #Iterate every minute
    time.sleep(60) 
    
    #Refresh pages
    driver1.refresh()  
    driver2.refresh()


