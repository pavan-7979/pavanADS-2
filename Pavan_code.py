#!/usr/bin/env python
# coding: utf-8

# In[4]:


# import all the important libraries. 
import pandas 
import matplotlib.pyplot as plt 
import numpy as np


# In[5]:


# creating function 
def read_data(Data_file):  
    co2=pandas.read_csv(Data_file,skiprows=4) # we are read dataset 
    co2_df = co2.fillna(co2.median(numeric_only=True))  # we are fill null valuse 
    co2_df2=co2_df.drop(['Country Code', 'Unnamed: 66', 'Indicator Name', 'Indicator Code'],axis=1)  #  we aer droping unuse columns 
    co2_df3=co2_df2.set_index("Country Name") # we are define index name
    co2_df3=co2_df3.T    # tranpose dataset                 
    co2_df3.reset_index(inplace=True) 
    co2_df3.rename(columns = {'index':'Year'}, inplace = True) 
    return co2_df,co2_df3   # return data frame 


# In[6]:


Data_file="/content/API_19_DS2_en_csv_v2_4700503.csv"  # we are read dataset path 

df1,df2=read_data(Data_file)  #  we are read two dataset frame 
df1.head()  # printing  dataset 


# In[7]:


df2.head() # printing df2 dataset 


# # Enter  Correlation for CO2 emissions from liquid fuel consumption (% oftotal)

# In[8]:


# Right now, we are compiling information for a report on "CO2 emission"
Dataset_corr = df1[df1['Indicator Name']=='CO2 emissions from liquid fuel consumption (% of total)']  
# At the moment, we are putting the finishing touches on a pivot table with the country names serving as the index and the years serving as the values.
Dataset_corr_1 = Dataset_corr.pivot_table(index=['Country Name'], values = ['2000', '2005', '2010', '2015', '2020'])  
# displaying the  10 sets of data with the symbol.
Dataset_corr_1 = Dataset_corr_1.head(7)  
Dataset_corr_1   


# # Time series  Analysis

# #Plotting  graph for Agricultural land (sq. km)

# In[9]:


# obtaining all of the information pertaining to Agricultural land
Dataframe = df1[df1['Indicator Name']=='Agricultural land (sq. km)']  
# index set.
Dataframe = Dataframe.set_index("Country Name") 
# dropping all the columns.
Dataframe = Dataframe.drop(['Country Code', 'Indicator Name', 'Indicator Code', 'Unnamed: 66'],axis=1)
Dataframe = Dataframe.T
# reset index.
Dataframe = Dataframe.reset_index() 


# In[10]:


# each and every variable that was used in the table
Dataframe_10=Dataframe.pivot_table(index=['index'], values=['Bangladesh','Italy','United Arab Emirates','Zambia','Angola','South Africa','European Union','Europe & Central Asia','Virgin Islands (U.S.)','Zimbabwe'])  
# define figure size 
plt.figure(figsize = (20,8)) 
plt.plot(Dataframe_10.head(15))
# set rotation of xticks. 
plt.xticks(rotation=45) 
# define all the legends.
plt.legend(['Bangladesh','Italy','United Arab Emirates','Zambia','Angola','South Africa','European Union','Europe & Central Asia','Virgin Islands (U.S.)','Zimbabwe']) 
plt.xlabel('Year')
#  x axis define year 
plt.ylabel('Comparison ')
# y axis define price 
plt.title('Agricultural land (sq. km)') 
# define title name 
plt.show() # showing graph 


# #plotting graph for Urban population (% of total population)

# In[11]:


# obtaining information regarding the Urban population
DATA_feame_10 = df1[df1['Indicator Name']=='Urban population (% of total population)']  
# we set index as country name.
DATA_feame_11 = DATA_feame_10.set_index("Country Name") 
# dropping all the unnecessary columns
DATA_feame_12=DATA_feame_11.drop(['Country Code', 'Indicator Name', 'Indicator Code', 'Unnamed: 66'],axis=1)
DATA_feame_13=DATA_feame_12.T
# reseting the index.
DATA_feame_14=DATA_feame_13.reset_index() 


# In[12]:


# We are selecting all data using pivot table like index and values.
dataframe_25=DATA_feame_14.pivot_table(index=['index'], values=['Bangladesh','Italy','United Arab Emirates','Zambia','Angola','South Africa','European Union','Europe & Central Asia','Virgin Islands (U.S.)','Zimbabwe'])  
# define figure size
plt.figure(figsize = (20,8))  
# print top 15 data.
plt.plot(dataframe_25.head(15))  
# set rotation for xticks.
plt.xticks(rotation=45) 
# define lagend.
plt.legend(['Bangladesh','Italy','United Arab Emirates','Zambia','Angola','South Africa','European Union','Europe & Central Asia','Virgin Islands (U.S.)','Zimbabwe'],bbox_to_anchor = (1.0,1.1),ncol=1) 
plt.xlabel('Year')
#  x axis define year 
plt.ylabel('Comparison ')
# y axis define price 
plt.title('Urban population (% of total population)  ') 
# define title name 
plt.show() # showing graph 


# # Pie Chart 

# In[13]:


# Extracting all the data for Foreign investment to make pie chart.
pie_data = df1[df1['Indicator Name']=='Foreign direct investment, net inflows (% of GDP)'] 


# In[14]:


# we picked up top 5 country with Foriegn investment using groupby. 
plt.subplot(2,2,1)
Top_5_Country = pie_data.groupby('Country Name')['1960'].max().sort_values(ascending = False).head(5)   
# passing all parameter for graph visualiation.            
Top_5_Country.plot(kind = "pie",figsize = (20 , 10),fontsize=12,  title="Top 5 Country With Foreign Investment(% of GDP)",autopct='%1.0f%%');                                                                                                    

plt.subplot(2,2,2) 
# we picked up top 5 country with Foriegn investment using groupby.
Top_5_Country = pie_data.groupby('Country Name')['1980'].max().sort_values(ascending = False).head(5)   
# passing all parameter for graph visualiation.            
Top_5_Country.plot(kind = "pie",figsize = (20 , 10),fontsize=12,  title="Top 5 Country With Foreign Investment(% of GDP)",autopct='%1.0f%%'); 

plt.subplot(2,2,3)
# we picked up top 5 country with Foriegn investment using groupby.
Top_5_Country = pie_data.groupby('Country Name')['2000'].max().sort_values(ascending = False).head(5)   
# passing all parameter for graph visualiation.            
Top_5_Country.plot(kind = "pie",figsize = (20 , 10),fontsize=12,  title="Top 5 Country With Foreign Investment(% of GDP)",autopct='%1.0f%%');                                                                                                   

plt.subplot(2,2,4)
# we picked up top 5 country with Foriegn investment using groupby.
Top_5_Country = pie_data.groupby('Country Name')['2020'].max().sort_values(ascending = False).head(5)   
# passing all parameter for graph visualiation.            
Top_5_Country.plot(kind = "pie",figsize = (20 , 10),fontsize=12,  title="Top 5 Country With Foreign Investment(% of GDP)",autopct='%1.0f%%'); 


# # Bar graph_1

# # Bar graph for Urban population

# In[15]:


# # Obtaining all of the information needed to create the bar graph for the urban population
Bar_Dataset = df1[df1['Indicator Name']=='Urban population']  
Bar_Dataset_1 = Bar_Dataset.pivot_table(index=['Country Name'], values = ['2000', '2005', '2010', '2015', '2020']) 
# showing top 10 country data. 
Bar_Dataset_1 = Bar_Dataset_1.head(10)  


# In[16]:


# define color, figure size fpor graph.
Bar_Dataset_1.plot.bar(color=['green','red','yellow','purple','orange'],figsize=(20,10))  
# set x_lable for graph
plt.xlabel('Country') 
# set y_label for graph.
plt.ylabel('Comparision')
# define title for graph.
plt.title('Urban population')  
plt.show(); 


# # Bar graph for Urban Population growth (annual %)

# In[17]:


# data extract for urban population growth (annual).
dataset_2 = df1[df1['Indicator Name']== 'Urban population growth (annual %)']  
# Using pivot table set index and values for making the graph.
dataset_bar_2 = dataset_2.pivot_table(index=['Country Name'], values = ['2000', '2005', '2010', '2015', '2020'])  
# Showing top 10 country growth year on year basis.
dataset_bar_2 = dataset_bar_2.head(10)  


# In[18]:


# define color and figure size for graph.
dataset_bar_2.plot.bar(color=['green','red','yellow','BlueViolet',  'Blue'],figsize=(20,10))  
# set x && y labels for graph.
plt.xlabel('Country') 
plt.ylabel('Comparision') 
# set title for graph.
plt.title('Urban population growth (annual %)')  
plt.show(); 


# # Creating Function for Extracting Data.

# In[19]:


# Right now, we are developing a function that will extract all of the data based on the country.
def func(country):
  heatmap_data_1 = df1[df1['Country Name']==f'{country}'] 
  heatmap_data_1 = heatmap_data_1.drop(['Country Name','Country Code','Indicator Code'],axis=1) 
  heatmap_data_1 = heatmap_data_1.T
  n_head_1=heatmap_data_1.iloc[0] 
  heatmap_data_1=heatmap_data_1[1:] 
  heatmap_data_1.columns=n_head_1
  heatmap_data_1 = heatmap_data_1.reset_index(drop=True)  
  return heatmap_data_1 


# # Heatmap (Afganistan) 

# In[20]:


# calling the function for 'Afganistan' Country.
Afghanistan_data = func("Afghanistan") 
# data save in new file.
Afghanistan_data.to_csv('new_dataset.csv')   


# In[21]:


# creating 
def read_dataset(Filename_1):
    afganistan=pandas.read_csv(Filename_1) 
    return afganistan


# In[22]:


Filename_new= '/content/new_dataset.csv' 
dataset= read_dataset(Filename_new) 
dataset = dataset.drop('Unnamed: 0',axis=1) 
dataset.head() 


# In[23]:


dataset = dataset[['Urban population (% of total population)','Urban population growth (annual %)','Nitrous oxide emissions (% change from 1990)','Methane emissions (% change from 1990)', 'Forest area (% of land area)', 'Forest area (sq. km)' ,'Access to electricity (% of population)']]  
corr=dataset.corr()
Arr=corr.to_numpy()           


# In[24]:


Label=dataset.columns 
import numpy as np 

fig, ax = plt.subplots(figsize=(15,8)) 
im = ax.imshow(corr)

# We want to show all ticks labels lenth 
ax.set_xticks(np.arange(len(Label)))
ax.set_yticks(np.arange(len(Label))) 
ax.set_xticklabels(Label)
ax.set_yticklabels(Label)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=60, ha="right",rotation_mode="anchor") 

# Loop over data dimensions and create text annotations.
for i in range(len(Label)):
    for j in range(len(Label)):
        text = ax.text(j, i, round(Arr[i, j],2), ha="center", va="center", color="w") 

ax.set_title("Afganistan") 
plt.show() 


# # Heatmap (Aruba)

# In[25]:


Aruba_data = func("Aruba") 
Aruba_data.to_csv('newd_data.csv')  


# In[26]:


def read_data_1(Filename):
    aruba=pandas.read_csv(Filename) 
    return aruba


# In[27]:


Filename= '/content/newd_data.csv' 
data= read_data_1(Filename) 
data = data.drop('Unnamed: 0',axis=1)
data.head()


# In[28]:


data = data[['Urban population (% of total population)','Urban population growth (annual %)','Nitrous oxide emissions (% change from 1990)','Methane emissions (% change from 1990)', 'Forest area (% of land area)', 'Forest area (sq. km)' ,'Access to electricity (% of population)']]  


# In[29]:


corr_1=data.corr() 

arr=corr_1.to_numpy() 


# In[30]:


Labels=data.columns
import numpy as np 

fig, ax = plt.subplots(figsize=(15,8)) 
im = ax.imshow(corr_1,cmap='copper')

# We want to show all ticks labels lenth
ax.set_xticks(np.arange(len(Labels)))
ax.set_yticks(np.arange(len(Labels)))
#label them with the respective list entries
ax.set_xticklabels(Labels)
ax.set_yticklabels(Labels)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=60, ha="right",rotation_mode="anchor") 

# Loop over data dimensions and create text annotations.
for i in range(len(Labels)):
    for j in range(len(Labels)):
        text = ax.text(j, i, round(arr[i, j],2), ha="center", va="center", color="w") 

ax.set_title("Aruba") 
plt.show() 


# In[30]:




