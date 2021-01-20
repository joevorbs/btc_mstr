import pandas as pd
import numpy as np
from selenium import webdriver
from datetime import datetime
from datetime import date
import time
import statsmodels.api as sm
import statistics 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.tsa.seasonal import STL
from scipy import signal
import seaborn as sns
import matplotlib.pyplot as plt

#Current date
today = datetime.now().date()

#Read in days data
btc_df = pd.read_csv("btc_mstr_data/btc/btc_data_" + str(today) + ".csv")
mstr_df = pd.read_csv("btc_mstr_data/mstr/mstr_data_" + str(today) + ".csv")

#Initialize scalers
std_scaler = StandardScaler()
minmax = MinMaxScaler()

#Append mstr price to bitcoin df to have both prices in one df
btc_df['mstr'] = mstr_df['price']

#Convert prices to proper data type
combined_df = btc_df[['price','mstr']]
combined_df['mstr'] = combined_df['mstr'].astype(float)
combined_df['price'] = combined_df['price'].str.replace(",","").astype(float)

#Log transform both values due to exponential movements in both assets 
combined_df['price'] = np.log(combined_df['price'])
combined_df['mstr'] = np.log(combined_df['mstr'])

#Standardize both values after taking their respective logs
log_std_prices = minmax.fit(combined_df[['price','mstr']]).transform(combined_df[['price','mstr']])

#Obtain hour+minute combinations and rename column
log_std_prices_df = pd.DataFrame(log_std_prices)
log_std_prices_df['time'] = btc_df['timestamp'].astype(str).str[0:5]
log_std_prices_df.rename(columns = {0: "btc", 1: "mstr"}, inplace = True)

#Univariate linear model - input is (y,x)
ols_log_std_prices = sm.OLS(log_std_prices_df["mstr"],log_std_prices_df["btc"]).fit()
#Obtain beta value and r2 
print("β1: " + str(ols_log_std_prices.params[0]))
print("R^2: " + str(ols_log_std_prices.rsquared))

#Plot standardized mstr & btc prices against each other - noticable patterns / trends?
fig, ax = plt.subplots() # Create the figure and axes object
log_std_prices_df.plot(x = 'time', y = 'btc', ax = ax) 
log_std_prices_df.plot(x = 'time', y = 'mstr', ax = ax, secondary_y = False)
plt.title("BTC and MSTR's Overlapped Price Movement\n " + "r = " + str(ols_log_std_prices.params[0]))
plt.xlabel('\nTime (Hour-Minute)')
plt.ylabel('\nPrice (Log Transformed & Standardized)')

#Polynomial regression - comparing fit
sns.regplot(log_std_prices_df['btc'] ,log_std_prices_df['mstr'], order=3)

#Pre-whiten data for cross correlation (time series as well later on)
#Seasonal trend decompisition using LOESS - obtain residuals for cross correlation
stl_btc = STL(tester2['btc'], period = len(tester2['btc']), seasonal=7)
res_btc = stl.fit()
#fig = res.plot()

stl_mstr = STL(tester2['mstr'], period = len(tester2['mstr']), seasonal=7)
res_mstr = stl.fit()

stl_btc_resid = res_btc.resid
stl_mstr_resid = res_mstr.resid
#fig = res.plot()

#Cross correlation plot - any significant lead or lag times?
plt.xcorr(stl_btc_resid, stl_mstr_resid, maxlags = 100)
plt.title("Cross Correlation between BTC and MSTR")
plt.xlabel("\nLead / Lag Times (Minutes)")
plt.ylabel("\nCorrelation Coefficient")

#Secondary cross correlation function - can see exact lead/lags
#def ccf(x, y, lag_max = 220):
        #result = ss.correlate(y - np.mean(y), x - np.mean(x), method = 'direct') / (np.std(y) * np.std(x) *len(y))
        #length = (len(result) - 1) // 2
        #lo = length - lag_max
        #hi = length + (lag_max + 1)
                  
        #return result[lo:hi]
        
#ccf_func_results = pd.DataFrame(ccf(log_std_prices_df['btc'],log_std_prices_df['mstr']), columns = ['ccf_values'])

#Print the lead/lags and their associated correlation coefficient
#for i, j in zip(ccf_func_results.index, ccf_func_results['ccf_values']):
    #print(i, j)

#Plot the CCF with 100 lead and lag times (minutes)
plt.xcorr(tester2['btc'], tester2['mstr'], maxlags = 100)

#BTC movement summary statistics
#Make a copy of the intial transformed df
btc_movement  = log_std_prices_df

#Offset the rows by 1 and subtract the value from the previous row
btc_movement['shifted_btc'] = btc_movement['btc'].shift(1)
btc_movement['difference_btc'] = btc_movement['shifted_btc'] - btc_movement['btc']
btc_movement['difference_btc'] = abs(btc_movement['difference_btc'])

#Summary stats
avg_btc = btc_movement['difference_btc'].mean()
max_btc = btc_movement['difference_btc'].max()
min_btc = btc_movement['difference_btc'].min()

btc_movement_summary = [avg_btc,max_btc,min_btc]

print("BTC Summary Stats")
for i,j in zip(['avg','max','min'], btc_movement_summary):
     print(i + ": ", j)

#MSTR movement statistics
copy_df = tester2
copy_df['shifted_mstr'] = copy_df['mstr'].shift(1)

copy_df['difference_mstr'] = copy_df['shifted_mstr'] - copy_df['mstr']

copy_df['difference_mstr'] = copy_df['difference_mstr'].abs()

avg_mstr = copy_df['difference_mstr'].mean()
max_mstr = copy_df['difference_mstr'].max()
min_mstr = copy_df['difference_mstr'].min()

print("BTC")
for i,j in zip(['avg','max','min'] ,[avg_btc,max_btc,min_btc]):
     print(i, j)