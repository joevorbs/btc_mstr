import pandas as pd
import numpy as np
from selenium import webdriver
from datetime import datetime
from datetime import date
import time
import statsmodels.api as sm
import statistics 
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

#Current date
today = datetime.now().date()

#Read in days data
btc_df = pd.read_csv("btc_mstr_data/btc/btc_data_" + str(today) + ".csv")
mstr_df = pd.read_csv("btc_mstr_data/mstr/mstr_data_" + str(today) + ".csv")

#Initialize scaler
scaler = StandardScaler()

#Append mstr price to bitcoin df to have both prices in one df
btc_df['mstr'] = mstr_df['price']
#Convert prices to proper type
tester = btc_df[['price','mstr']]
tester['mstr'] = tester['mstr'].astype(float)
tester['price'] = tester['price'].str.replace(",","").astype(float)
#Standardize prices
joe = scaler.fit(tester[['price','mstr']]).transform(tester[['price','mstr']])

#Obtain hour+minute combinations and rename column
tester2 = pd.DataFrame(joe)
tester2['time'] = btc_df['timestamp'].astype(str).str[0:5]
tester2.rename(columns = {0: "btc", 1: "mstr"}, inplace = True)

#Univariate linear model - input is (y,x)
results = sm.OLS(tester2["mstr"],tester2["btc"]).fit()
results.summary()

#Plot standardized mstr & btc prices against each other - noticable patterns / trends?
fig, ax = plt.subplots() # Create the figure and axes object
tester2.plot(x = 'time', y = 'btc', ax = ax) 
tester2.plot(x = 'time', y = 'mstr', ax = ax, secondary_y = True)

#Polynomial regression - comparing fit
sns.regplot(tester2['btc'] ,tester2['mstr'], order=3)

#Cross correlation function - any lead/lag times in terms of minutes?
def ccf(x, y, lag_max = 100):
        result = ss.correlate(y - np.mean(y), x - np.mean(x), method = 'direct') / (np.std(y) * np.std(x) *len(y))
        length = (len(result) - 1) // 2
        lo = length - lag_max
        hi = length + (lag_max + 1)
                  
        return result[lo:hi]

#Create a dataframe of the cross correlation values and sort them
joey = pd.DataFrame(ccf(tester2['btc'], tester2['mstr'])).sort_values(by=0, ascending = True)
joey.rename(columns = {0: "ccf"}, inplace = True)

#Print the position of the indicator and its associated correlation coefficient - to eyeball
for i, j in zip(joey.index, joey['ccf']):
    print(i, j)

#Plot the CCF with 100 lead and lag times (minutes)
plt.xcorr(tester2['btc'], tester2['mstr'], maxlags = 100)