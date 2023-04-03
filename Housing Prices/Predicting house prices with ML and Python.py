#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

##read them in a list comprehension/ parse any date

fed_files = ["MORTGAGE30US.csv", "RRVRUSQ156N.csv", "CPIAUCSL.csv"]

dfs= [pd.read_csv(f,parse_dates=True, index_col=0) for f in fed_files]


# In[2]:


dfs[0]


# In[3]:


dfs[1]


# In[4]:


dfs[2]


# In[5]:


fed_data= pd.concat(dfs, axis=1)


# In[6]:


fed_data


# In[7]:


fed_data.tail(50)


# In[8]:


fed_data = fed_data.ffill()


# In[9]:


fed_data.tail(50)


# In[10]:


zillow_files=["Metro_median_sale_price_uc_sfrcondo_week.csv", "Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_month.csv"]

dfs = [pd.read_csv(f) for f in zillow_files]


# In[11]:


dfs[0]


# In[12]:


##select only first row and 5th column onwards the rest drop it
dfs=[pd.DataFrame(df.iloc[0,5:]) for df in dfs]


# In[13]:


dfs[0]


# In[14]:


dfs[1]


# In[15]:


for df in dfs:
    df.index = pd.to_datetime(df.index)
    df["month"]= df.index.to_period("M")


# In[16]:


dfs[0]


# In[17]:


dfs[1]


# In[18]:


price_data = dfs[0].merge(dfs[1], on="month")


# In[19]:


price_data.index = dfs[0].index


# In[20]:


price_data


# In[21]:


del price_data["month"]
price_data.columns = ["price","value"]


# In[22]:


price_data


# In[23]:


fed_data = fed_data.dropna()


# In[24]:


fed_data.tail(20)


# In[25]:


from datetime import timedelta

fed_data.index = fed_data.index + timedelta(days=2)


# In[26]:


fed_data


# In[27]:


price_data = fed_data.merge(price_data, left_index=True, right_index=True)


# In[28]:


price_data


# In[29]:


price_data.columns = ["interest", 'vacancy', 'cpi', 'price','value']


# In[30]:


price_data


# In[31]:


price_data.plot.line(y="price", use_index=True)


# In[32]:


price_data["adj_price"]=price_data["price"]/price_data["cpi"]*100


# In[33]:


price_data.plot.line(y="adj_price", use_index=True)


# In[34]:


price_data["adj_value"]=price_data["value"]/price_data["cpi"]*100


# In[35]:


price_data["next_quarter"]=price_data["adj_price"].shift(-13)


# In[36]:


price_data


# In[37]:


price_data.dropna(inplace=True)


# In[38]:


price_data


# In[39]:


price_data["change"]=(price_data["next_quarter"]>price_data["adj_price"]).astype(int)


# In[40]:


price_data


# In[41]:


price_data["change"].value_counts()


# In[42]:


predictors = ["interest", "vacancy","adj_price","adj_value"]
target="change"


# In[43]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score #% of time model is correct
import numpy as np


# In[44]:


START = 260
STEP = 52

def predict(train, test, predictors, target):
    rf = RandomForestClassifier(min_samples_split=10, random_state=1)
    rf.fit(train[predictors], train[target])
    preds = rf.predict(test[predictors])
    return preds

def backtest(data, predictos, target):
    all_preds=[]
    for i in range(START, data.shape[0], STEP):
        train= price_data.iloc[:i]
        test = price_data.iloc[i:(i+STEP)]
        all_preds.append(predict(train, test, predictors, target))
        
    preds = np.concatenate(all_preds)
    return preds, accuracy_score(data.iloc[START:][target],preds)


# In[45]:


preds, accuracy = backtest(price_data, predictors,target)


# In[46]:


accuracy


# In[47]:


yearly=price_data.rolling(52,min_periods=1).mean()


# In[48]:


yearly_ratios=[p+"_year" for p in predictors]
price_data[yearly_ratios]=price_data[predictors]/yearly[predictors]


# In[49]:


price_data


# In[50]:


preds, accuracy = backtest(price_data, predictors + yearly_ratios, target)


# In[51]:


accuracy


# In[52]:


pred_match = (preds==price_data[target].iloc[START:])


# In[53]:


pred_match[pred_match == True]="green"
pred_match[pred_match == False]="red"


# In[54]:


import matplotlib.pyplot as plt

plot_data = price_data.iloc[START:].copy()

plot_data.reset_index().plot.scatter(x="index", y="adj_price", color=pred_match)


# In[55]:


from sklearn.inspection import permutation_importance

rf = RandomForestClassifier(min_samples_split=10, random_state=10)
rf.fit(price_data[predictors], price_data[target])

result = permutation_importance(rf,price_data[predictors],price_data[target],n_repeats=10, random_state=1)


# In[56]:


result["importances_mean"]


# In[57]:


predictors


# In[ ]:




