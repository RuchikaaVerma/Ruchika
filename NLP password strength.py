#!/usr/bin/env python
# coding: utf-8

# In[15]:


## importing all necessary libraries ..

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[16]:


import sqlite3


# In[20]:


con = sqlite3.connect(r"C:\Users\Ruchika Verma\Downloads\password_resources\password_Data.sqlite")


# In[21]:


data= pd.read_sql_query("SELECT * FROM Users" ,con)


# In[22]:


data.shape


# In[23]:


data.head(4)


# In[24]:


#Doing Basic DATA  Cleaning!
data.columns


# In[25]:


data.drop(["index"] , axis=1 , inplace=True)


# In[26]:


data.head(4)


# In[27]:


#checking duplicate data
data.duplicated().sum()


# In[28]:


#check the missing values in the columns
data.isnull().any()


# In[29]:


data.isnull().any().sum() ## it means 0 feature have NAN values


# In[30]:


data.dtypes


# In[31]:


data["strength"]


# In[32]:


data["strength"].unique()


# In[33]:


#3.performing sematic analysis!
data.columns


# In[34]:


data["password"]


# In[35]:


data["password"][0]


# In[36]:


type(data["password"][0])


# In[37]:


data["password"].str.isnumeric()#numeric character only


# In[38]:


data[data["password"].str.isnumeric()]


# In[42]:


data[data["password"].str.isnumeric()].shape 


# In[43]:


data[data["password"].str.isupper()]


# In[44]:


data[data["password"].str.isalpha()].shape

### around 50 users have their password as alphabet letters only !


# In[45]:


data[data["password"].str.isalnum()]


# In[46]:


data[data["password"].str.istitle()]

### around 932 users have their password having first alphabet capital !


# In[47]:


data["password"]


# In[48]:


import string


# In[49]:


string.punctuation ## all punctuations defined in "string" package !


# In[50]:


def find_semantics(row):
    for char in row:
        if char in string.punctuation:
            return 1
        else:
            pass


# In[51]:


data["password"].apply(find_semantics)==1


# In[52]:


data[data["password"].apply(find_semantics)==1]

## ie , 2663 observations have special characters in between them ..
## 2.6% people password actually uses special character in their password ..


# In[53]:


#lengthof every password
data["password"][0]


# In[54]:


len(data["password"][0]) 


# In[55]:


data["length"] = data["password"].str.len() 


# In[56]:


password = "Shan99" #frequency of lower case


# In[57]:


[char for char in password if char.islower()]


# In[58]:


len([char for char in password if char.islower()])


# In[59]:


len([char for char in password if char.islower()])/len(password)


# In[60]:


def freq_lowercase(row):
    return len([char for char in row if char.islower()])/len(row)


# In[61]:


def freq_uppercase(row):
    return len([char for char in row if char.isupper()])/len(row)#frequency of uppercase 


# In[62]:


def freq_numerical_case(row):
    return len([char for char in row if char.isdigit()])/len(row)


# In[63]:


data["lowercase_freq"] = np.round(data["password"].apply(freq_lowercase) , 3)

data["uppercase_freq"] = np.round(data["password"].apply(freq_uppercase) , 3)

data["digit_freq"] = np.round(data["password"].apply(freq_numerical_case) , 3)


# In[64]:


data.head(3)


# In[65]:


def freq_special_case(row):
    special_chars = []
    for char in row:
        if not char.isalpha() and not char.isdigit():
            special_chars.append(char)
    return len(special_chars)


# In[66]:


data["special_char_freq"] = np.round(data["password"].apply(freq_special_case) , 3) ## applying "freq_special_case" function


# In[67]:


data.head(5)


# In[68]:


data["special_char_freq"] = data["special_char_freq"]/data["length"] ## noromalising "special_char_freq" feature 


# In[69]:


data.head(5)


# In[70]:


data.columns#performing statistics
data.columns


# In[71]:


data[['length' , 'strength']].groupby(['strength']).agg(["min", "max" , "mean" , "median"])


# In[72]:


cols = ['length', 'lowercase_freq', 'uppercase_freq',
       'digit_freq', 'special_char_freq']

for col in cols:
    print(col)
    print(data[[col , 'strength']].groupby(['strength']).agg(["min", "max" , "mean" , "median"]))
    print('\n')


# In[73]:


data.columns


# In[74]:


fig , ((ax1 , ax2) , (ax3 , ax4) , (ax5,ax6)) = plt.subplots(3 , 2 , figsize=(15,7))

sns.boxplot(x="strength" , y='length' , hue="strength" , ax=ax1 , data=data)
sns.boxplot(x="strength" , y='lowercase_freq' , hue="strength" , ax=ax2, data=data)
sns.boxplot(x="strength" , y='uppercase_freq' , hue="strength" , ax=ax3, data=data)
sns.boxplot(x="strength" , y='digit_freq' , hue="strength" , ax=ax4, data=data)
sns.boxplot(x="strength" , y='special_char_freq' , hue="strength" , ax=ax5, data=data)

plt.subplots_adjust(hspace=0.6)


# In[75]:


data.columns
#how to short imp.feature


# In[76]:


def get_dist(data , feature):
    
    plt.figure(figsize=(10,8))
    plt.subplot(1,2,1)
    
    sns.violinplot(x='strength' , y=feature , data=data )
    
    plt.subplot(1,2,2)
    
    sns.distplot(data[data['strength']==0][feature] , color="red" , label="0" , hist=False)
    sns.distplot(data[data['strength']==1][feature], color="blue", label="1", hist=False)
    sns.distplot(data[data['strength']==2][feature], color="orange", label="2", hist=False)
    plt.legend()
    plt.show()


# In[77]:


import warnings 
from warnings import filterwarnings
filterwarnings("ignore")


# In[78]:


get_dist(data , "length")


# In[79]:


data.columns


# In[80]:


get_dist(data , 'lowercase_freq')


# In[81]:


get_dist(data , 'uppercase_freq')


# In[82]:


get_dist(data , 'digit_freq')


# In[83]:


get_dist(data , 'special_char_freq')


# In[84]:


#applying tf_idf on data
data.head(4)


# In[85]:


data


# In[86]:


dataframe = data.sample(frac=1) ### shuffling randomly for robustness of ML moodel 


# In[87]:


dataframe


# In[88]:


x = list(dataframe["password"])


# In[89]:


from sklearn.feature_extraction.text import TfidfVectorizer ## import TF-IDF vectorizer to convert text data into numerical data


# In[90]:


vectorizer = TfidfVectorizer(analyzer="char")


# In[91]:


X = vectorizer.fit_transform(x)


# In[92]:


X.shape


# In[93]:


dataframe["password"].shape


# In[94]:


X


# In[95]:


X.toarray()  ### to get entire matrix of TF-IDF for 100000 passwords ..


# In[96]:


X.toarray()[0] ## TF-IDF scores of Ist row


# In[97]:


dataframe["password"]


# In[98]:


len(vectorizer.get_feature_names_out())


# In[99]:


### returns feature/char_of_passwords/columns names

vectorizer.get_feature_names_out()

## ie these are the various chars to which different TF-IDF values are assigned for 100000 passwords ..


# In[100]:


df2 = pd.DataFrame(X.toarray() , columns=vectorizer.get_feature_names_out())


# In[101]:


df2


# In[102]:


#applying ml alogrithm
dataframe.columns


# In[103]:


df2["length"] = dataframe['length']
df2["lowercase_freq"] = dataframe['lowercase_freq']


# In[104]:


df2


# In[105]:


y = dataframe["strength"]


# In[106]:


from sklearn.model_selection import train_test_split


# In[107]:


X_train, X_test, y_train, y_test = train_test_split(df2, y, test_size=0.20)


# In[108]:


X_train.shape


# In[109]:


y_train.shape


# In[110]:


from sklearn.linear_model import LogisticRegression


# In[111]:


## Apply Multinomial logistic Regression as have data have 3 categories in outcomes

clf = LogisticRegression(multi_class="multinomial")


# In[112]:


clf.fit(X_train , y_train)


# In[113]:


y_pred = clf.predict(X_test) ## doing prediction on X-Test data


# In[114]:


y_pred


# In[115]:


from collections import Counter


# In[116]:


Counter(y_pred)


# In[117]:


#prediction
password = "%@123abcd"


# In[118]:


sample_array = np.array([password])


# In[119]:


sample_matrix = vectorizer.transform(sample_array)


# In[120]:


sample_matrix.toarray()


# In[121]:


sample_matrix.toarray().shape

### right now , array dim. is (1,99) so now we need to make it as : (1,101) so that my model will accept it as input..
### ie we need to add (length_of_password) & (total_lowercase_chars) in passsword


# In[122]:


password


# In[123]:


len(password)


# In[124]:


[char for char in password if char.islower()]


# In[125]:


len([char for char in password if char.islower()])/len(password)


# In[126]:


np.append(sample_matrix.toarray() , (9,0.444)).shape


# In[127]:


np.append(sample_matrix.toarray() , (9,0.444)).reshape(1,101)


# In[128]:


np.append(sample_matrix.toarray() , (9,0.444)).reshape(1,101).shape


# In[129]:


new_matrix = np.append(sample_matrix.toarray() , (9,0.444)).reshape(1,101)


# In[130]:


clf.predict(new_matrix)


# In[131]:


def predict():
    password = input("Enter a password : ")
    sample_array = np.array([password])
    sample_matrix = vectorizer.transform(sample_array)
    
    length_pass = len(password)
    length_normalised_lowercase = len([char for char in password if char.islower()])/len(password)
    
    new_matrix2 = np.append(sample_matrix.toarray() , (length_pass , length_normalised_lowercase)).reshape(1,101)
    result = clf.predict(new_matrix2)
    
    if result == 0 :
        return "Password is weak"
    elif result == 1 :
        return "Password is normal"
    else:
        return "password is strong"


# In[132]:


predict()


# In[133]:


#model evalution
from sklearn.metrics import confusion_matrix ,  accuracy_score , classification_report


# In[134]:


accuracy_score(y_test , y_pred)


# In[135]:


confusion_matrix(y_test , y_pred)


# In[136]:


print(classification_report(y_test , y_pred))# model report


# In[ ]:




