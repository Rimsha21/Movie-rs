
# coding: utf-8

# # ITEM BASED COLLABORATIVE FILTERING - MOVIE RECOMMENDATION SYSTEM BY RIMSHA V.

# In[2]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# In[3]:


my_df=pd.read_csv("ratings.csv")


# In[4]:


my_df.head()


# In[5]:


movie_titles=pd.read_csv("movies.csv")
movie_titles.head()


# In[9]:


my_df=pd.merge(my_df,movie_titles,on='movieId')
my_df.tail()


# In[11]:


columns=['title_x','title_y','genres_x','genres_y','genres']
my_df.drop(columns,axis=1,inplace=True)


# In[12]:


my_df.head()


# In[13]:


my_df.describe()


# In[14]:


rating = pd.DataFrame(my_df.groupby('title')['rating'].mean())
rating.head()


# In[15]:


rating['number of ratings']=my_df.groupby('title')['rating'].count()
rating.head()


# In[16]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[17]:


sns.jointplot(x='rating',y='number of ratings',data=rating)


# # MOVIE MATRIX

# In[20]:


movie_matrix_UII= my_df.pivot_table(index='userId',columns='title',values='rating')
movie_matrix_UII.tail()


# # MOST RATED MOVIES

# In[21]:


rating.sort_values('number of ratings',ascending=False).head()


# # Making a recommendation for Forrest Gump

# In[26]:


Forrest_gump_rating = movie_matrix_UII['Forrest Gump (1994)']
similar = movie_matrix_UII.corrwith(Forrest_gump_rating)
similar.head()


# In[27]:


corr_fg= pd.DataFrame(similar,columns=['Correlation'])
corr_fg.dropna(inplace=True)
corr_fg.head()


# In[28]:


corr_fg= corr_fg.join(rating['number of ratings'])
corr_fg.head()


# In[29]:


corr_fg[corr_fg['number of ratings']>20].sort_values(by='Correlation',ascending=False).head()


# # 
