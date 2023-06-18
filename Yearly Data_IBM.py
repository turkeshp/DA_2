#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#EDA


# In[2]:



import time
import random
import numpy as np
import pandas as pd
import datetime as datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
#from fbprophet import Prophet
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from matplotlib import animation, rc
rc('animation', html='jshtml')
import seaborn as sns
from pylab import rcParams
from datetime import datetime
from datetime import date
from datetime import timedelta
from scipy.stats import probplot
#from fbprophet import Prophet


# In[3]:


import pandas as pd

# Update the file path by escaping the backslashes
file_path = 'C:\\Users\\Admin\\Downloads\\Data IBM\\2 Year IBM Stock Data.csv'

# Read the CSV file into a DataFrame
data0 = pd.read_csv(file_path)

# Reverse the order of rows and reset the index
data = data0[::-1].reset_index(drop=True)


# In[4]:


print(data.head(20))


# In[5]:


data.columns=['date','open','high','low','close','volume']
data['60-min mean']=data['close'].rolling(window=60).mean()
data['60-min std']=data['close'].rolling(window=60).std()
data['20-min mean']=data['close'].rolling(window=20).mean()
data['20-min std']=data['close'].rolling(window=20).std()
display(data)


# In[6]:


fig=make_subplots(specs=[[{"secondary_y":False}]])
fig.add_trace(go.Scatter(x=data['date'][-100:],y=data['close'][-100:],name='close'),secondary_y=False,)
fig.add_trace(go.Scatter(x=data['date'][-100:],y=data['60-min mean'][-100:],name='60-min mean'),secondary_y=False,)
fig.add_trace(go.Scatter(x=data['date'][-100:],y=data['60-min mean'][-100:]+2*data['60-min std'][-100:],name='60-min mean+2std'),secondary_y=False,)
fig.add_trace(go.Scatter(x=data['date'][-100:],y=data['60-min mean'][-100:]-2*data['60-min std'][-100:],name='60-min mean-2std'),secondary_y=False,)
fig.update_layout(autosize=False,width=800,height=400,title_text='IBM 60-min')
fig.update_xaxes(title_text="Date")
fig.update_yaxes(title_text="USD",secondary_y=False)
fig.show()

fig=make_subplots(specs=[[{"secondary_y":False}]])
fig.add_trace(go.Scatter(x=data['date'][-100:],y=data['close'][-100:],name='close'),secondary_y=False,)
fig.add_trace(go.Scatter(x=data['date'][-100:],y=data['20-min mean'][-100:],name='20-min mean'),secondary_y=False,)
fig.add_trace(go.Scatter(x=data['date'][-100:],y=data['20-min mean'][-100:]+2*data['20-min std'][-100:],name='20-min mean+2std'),secondary_y=False,)
fig.add_trace(go.Scatter(x=data['date'][-100:],y=data['20-min mean'][-100:]-2*data['20-min std'][-100:],name='20-min mean-2std'),secondary_y=False,)
fig.update_layout(autosize=False,width=800,height=400,title_text='IBM 20-min')
fig.update_xaxes(title_text="Date")
fig.update_yaxes(title_text="USD",secondary_y=False)
fig.show()


# In[7]:


import pandas as pd
import matplotlib.pyplot as plt

# Update the file path by escaping the backslashes
file_path = 'C:\\Users\\Admin\\Downloads\\Data IBM\\2 Year IBM Stock Data.csv'

# Read the CSV file into a DataFrame
data0 = pd.read_csv(file_path)

# Reverse the order of rows and reset the index
data = data0[::-1].reset_index(drop=True)

# Rename the column 'Date' to 'time' if needed
data.rename(columns={'Date': 'time'}, inplace=True)

# Convert the 'time' column to datetime format
data['time'] = pd.to_datetime(data['time'])

# Perform insights and EDA on the data

# Display first few rows of the dataset
print(data.head())

# Check data types and missing values
print(data.info())

# Generate descriptive statistics of numerical columns
print(data.describe())

# Plot stock prices over time
plt.figure(figsize=(12, 6))
plt.plot(data['time'], data['close'])
plt.title('IBM Stock Prices Over Time')
plt.xlabel('Time')
plt.ylabel('Closing Price')
plt.show()

# Plot distribution of stock prices
plt.figure(figsize=(8, 6))
plt.hist(data['close'], bins=20)
plt.title('Distribution of IBM Stock Prices')
plt.xlabel('Closing Price')
plt.ylabel('Frequency')
plt.show()

# Calculate and plot moving averages
data['MA_50'] = data['close'].rolling(window=50).mean()
data['MA_200'] = data['close'].rolling(window=200).mean()

plt.figure(figsize=(12, 6))
plt.plot(data['time'], data['close'], label='Closing Price')
plt.plot(data['time'], data['MA_50'], label='50-day Moving Average')
plt.plot(data['time'], data['MA_200'], label='200-day Moving Average')
plt.title('IBM Stock Prices with Moving Averages')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# Calculate and plot trading volume
plt.figure(figsize=(12, 6))
plt.plot(data['time'], data['volume'])
plt.title('IBM Trading Volume Over Time')
plt.xlabel('Time')
plt.ylabel('Volume')
plt.show()

# Calculate correlation matrix
correlation_matrix = data[['open', 'high', 'low', 'close', 'volume']].corr()

# Plot correlation matrix as a heatmap
plt.figure(figsize=(8, 6))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.title('Correlation Matrix of IBM Stock Data')
plt.xticks(range(correlation_matrix.shape[1]), correlation_matrix.columns, rotation=45)
plt.yticks(range(correlation_matrix.shape[1]), correlation_matrix.columns)
plt.show()


# In[8]:


import pandas as pd

# Update the file path by escaping the backslashes
file_path = 'C:\\Users\\Admin\\Downloads\\Data IBM\\2 Year IBM Stock Data.csv'

# Read the CSV file into a DataFrame
data0 = pd.read_csv(file_path)

# Reverse the order of rows and reset the index
data = data0[::-1].reset_index(drop=True)

# Convert the 'time' column to datetime format
data['time'] = pd.to_datetime(data['time'])

# Change the date format to a similar one as in Excel
data['time'] = data['time'].dt.strftime('%m-%d-%Y %H:%M')

# Display the updated DataFrame
print(data.head())


# In[9]:


import pandas as pd

# Update the file path by escaping the backslashes
file_path = 'C:\\Users\\Admin\\Downloads\\Data IBM\\monthly_Divi.csv'

# Read the CSV file into a DataFrame
data0 = pd.read_csv(file_path)

# Reverse the order of rows and reset the index
data = data0[::-1].reset_index(drop=True)


# In[10]:


print(data.head(20))


# In[11]:


import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
data = pd.read_csv('C:\\Users\\Admin\\Downloads\\Data IBM\\monthly_Divi.csv')

# Convert the 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Set the 'Date' column as the index
data.set_index('Date', inplace=True)

# Perform EDA and visualization
# Example: Plotting the closing price over time
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'])
plt.title('IBM Stock Closing Price')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[30]:


import pandas as pd

# Update the file path by escaping the backslashes
file_path = 'C:\\Users\\Admin\\Downloads\\Data IBM\\weekly_data.csv'

# Read the CSV file into a DataFrame
data0 = pd.read_csv(file_path)

# Reverse the order of rows and reset the index
data = data0[::-1].reset_index(drop=True)


# In[33]:


print(data.head(200))


# In[38]:



# Resample the data on a weekly basis and calculate the weekly statistics
weekly_data = data.resample('W').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
})

# Perform stock analysis on the weekly data
# Example: Calculate weekly returns
weekly_data['Returns'] = weekly_data['Close'].pct_change()

# Print the weekly data and analysis results
print(weekly_data)


# In[39]:


weekly_data['Returns'] = weekly_data['Close'].pct_change()


# In[40]:


weekly_data['Range'] = weekly_data['High'] - weekly_data['Low']


# In[43]:


# Calculate the average weekly volume
weekly_data['Average Volume'] = weekly_data['Volume'] / 5

# Plot the weekly volume trends
weekly_data['Volume'].plot(kind='bar', figsize=(10, 6), title='Weekly Volume Trends')
plt.xlabel('Week')
plt.ylabel('Volume')
plt.show()


# In[ ]:





# In[ ]:





# In[42]:





# In[ ]:




