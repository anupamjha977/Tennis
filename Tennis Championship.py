#!/usr/bin/env python
# coding: utf-8

# # Tennis Championship 2016 2017
# **Life Cycle of Exploratory Data Analysis**
# 
# * Data Collection
# 
# * Data Cleaning
# 
# * Descriptive Statistics.
# 
# * Data Visualization

# **OBJECTIVE**
#  We are going to the analysis of Tennis Championship Title from 2016 to 2017
#  

# ## 1 Data Collection
# 
# * The Dataset is collected from this link https://www.kaggle.com/datasets/serangu/woman-tennis-2016-2017-years
# 
# * **Context:** The dataset contains more than 85,000 women's tennis matches that took place in 2016-2017.
# 

# ## 1.1 importing the required file

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings

warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')
#Display all the columns of the dataframe
pd.pandas.set_option("display.max_columns", None)


# ## 1.2 storing the file in pandas dataframe

# In[2]:


df=pd.read_csv(r"E:\2016-2017_orig.csv",sep=";")#create the data frame


# In[3]:


df.shape #getting the data set shape rows and column


# In[4]:


df.head()


# ## 2) Data Cleaning

# ### 3.1 Handling missing values and other anomolies

# In[5]:


df.isnull().sum() #no null value in data


# In[6]:


df.dtypes


# ### check unique value of column

# In[7]:


df["name_1"].value_counts()


# In[8]:


df["player1_is_win?"].value_counts() #equally distributed


# ### check for duplicate value

# In[9]:


df.duplicated().sum()


# ## 3. Descriptive Statistics

# In[10]:


df.describe()


# In[11]:


df.corr() # finding the correlation between all values


# In[12]:


df.cov() #covariance matrix


# In[13]:


df.skew()


# ## 4.  visualization

# ### 4.1 univariate analysis

# ### distplot

# In[14]:


num_col=[fe for fe in df.columns if df[fe].dtype!="O"] #finding the numerical column


# In[15]:


num_col


# In[16]:


cat_col=[fe for fe in df.columns if df[fe].dtype=="O"] #finding the categorical column


# In[17]:


cat_col


# In[18]:


#distplot to check skewness
plt.figure(figsize=(15, 15))
plt.suptitle('Univariate Analysis of Numerical Features', fontsize=20, fontweight='bold', alpha=0.8, y=1.)
for i in range(0,len(num_col)):
    plt.subplot(3, 3, i+1)
    sns.distplot(df[num_col[i]], color='g')
    plt.xlabel(num_col[i])
    plt.tight_layout()


# In[19]:


## Boxplot to find Outliers in the features
plt.figure(figsize=(15, 15))
plt.suptitle('Univariate Analysis of Numerical Features using box plot', fontsize=20, fontweight='bold', alpha=0.8, y=1.)
sns.boxplot(data = df,orient="v")


# In[20]:


player_wins = df.groupby('name_1')['player1_is_win?'].sum()
player_wins.sort_values(ascending=False)


# In[21]:


import plotly.graph_objects as go

# Assuming the given DataFrame is named 'df'
player_wins = df.groupby('name_1')['player1_is_win?'].sum()
top_10_players = player_wins.sort_values(ascending=False).head(10)

# Create the bar chart using Plotly
fig = go.Figure(data=[go.Bar(x=top_10_players.index, y=top_10_players)])

# Set the chart title and axes labels
fig.update_layout(
    title="Top 10 Players with the Most Wins",
    xaxis_title="Player",
    yaxis_title="Number of Wins"
)

# Show the chart
fig.show()


# In[22]:


import plotly.graph_objects as go

# Assuming the given DataFrame is named 'df'
player_wins = df.groupby('name_1')['player1_is_win?'].sum()
player_matches = df.groupby('name_1')['player1_is_win?'].count()
player_win_percentage = (player_wins / player_matches) * 100
top_10_players = player_win_percentage.sort_values(ascending=False).head(10)

# Create the bar chart using Plotly
fig = go.Figure(data=[go.Bar(x=top_10_players.index, y=top_10_players)])

# Set the chart title and axes labels
fig.update_layout(
    title="Top 10 Players with the Highest Winning Percentages",
    xaxis_title="Player",
    yaxis_title="Winning Percentage"
)

# Show the chart
fig.show()


# In[24]:


import pandas as pd

# Concatenate the 'name_1' and 'name_2' columns to create pairs
pairs = df[['name_1', 'name_2']].apply(sorted, axis=1).apply(' - '.join)

# Count the occurrences of each pair
pair_counts = pairs.value_counts()

# Remove the pairs where the order of names is reversed
pair_counts = pair_counts[~pair_counts.index.duplicated(keep='first')]

# Get the top 10 most frequent pairs
top_10_pairs = pair_counts.head(10)

print(top_10_pairs)


# In[ ]:




